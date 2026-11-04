use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use tracing::{info, instrument};

use crate::types::{TodoItem, TodoState, TodoStatus, parse_toon_table};

use super::ToonTool;

pub struct TodoTool {
    state: Arc<Mutex<TodoState>>,
}

impl TodoTool {
    pub fn new(state: Arc<Mutex<TodoState>>) -> Self {
        Self { state }
    }
}

#[async_trait]
impl ToonTool for TodoTool {
    fn name(&self) -> &'static str {
        "todo_write"
    }

    fn description(&self) -> &'static str {
        "Replace the active task list and update statuses. Send complete TOON rows or JSON with a tasks array; status values are Pending, InProgress, Completed, or Failed."
    }

    fn parameters_signature(&self) -> &'static str {
        "todo_write[N]{id,description,status}:"
    }

    fn result_signature(&self) -> &'static str {
        "tasks[N]{id,description,status}:"
    }

    #[instrument(name = "tool.todo.write", skip(self, payload))]
    async fn execute(&self, payload: &str) -> Result<String> {
        let rows = parse_toon_table(payload, self.name(), &["id", "description", "status"])?;
        let mut ids = HashSet::with_capacity(rows.len());
        let tasks = rows
            .into_iter()
            .map(|row| {
                let id = row[0]
                    .parse::<usize>()
                    .context("Todo id must be an integer")?;
                if id == 0 {
                    bail!("Todo id must be greater than zero");
                }
                if !ids.insert(id) {
                    bail!("Duplicate todo id {id}");
                }
                if row[1].trim().is_empty() {
                    bail!("Todo description must not be empty");
                }
                Ok(TodoItem {
                    id,
                    description: row[1].clone(),
                    status: row[2].parse::<TodoStatus>()?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let mut state = self
            .state
            .lock()
            .map_err(|error| anyhow::anyhow!("Todo state mutex is poisoned: {error}"))?;

        for task in &tasks {
            if let Some(previous) = state.tasks.iter().find(|old| old.id == task.id)
                && previous.status != task.status
            {
                info!(
                    task_id = task.id,
                    from = %previous.status,
                    to = %task.status,
                    "Todo task status changed"
                );
            }
        }

        state.tasks = tasks;
        Ok(state.render_toon_context())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use crate::types::{TodoState, TodoStatus};

    use super::{TodoTool, ToonTool};

    #[tokio::test]
    async fn updates_shared_todo_state_and_returns_refreshed_toon() {
        let state = Arc::new(Mutex::new(TodoState::default()));
        let tool = TodoTool::new(Arc::clone(&state));

        let initial = tool
            .execute("todo_write[1]{id,description,status}:\n1,Audit file tree,Pending")
            .await
            .unwrap();
        assert_eq!(
            initial,
            "tasks[1]{id,description,status}:\n1,Audit file tree,Pending"
        );

        let updated = tool
            .execute("todo_write[1]{id,description,status}:\n1,Audit file tree,InProgress")
            .await
            .unwrap();
        assert!(updated.ends_with("1,Audit file tree,InProgress"));
        assert_eq!(
            state.lock().unwrap().tasks[0].status,
            TodoStatus::InProgress
        );

        let completed = tool
            .execute("todo_write[1]{id,description,status}:\n1,Audit file tree,Completed")
            .await
            .unwrap();
        assert!(completed.ends_with("1,Audit file tree,Completed"));
    }

    #[tokio::test]
    async fn rejects_duplicate_ids_without_changing_state() {
        let state = Arc::new(Mutex::new(TodoState::default()));
        let tool = TodoTool::new(Arc::clone(&state));

        let result = tool
            .execute("todo_write[2]{id,description,status}:\n1,First,Pending\n1,Second,Pending")
            .await;
        assert!(result.is_err());
        assert!(state.lock().unwrap().tasks.is_empty());
    }
}
