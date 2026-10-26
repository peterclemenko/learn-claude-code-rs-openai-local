{ inputs, ... }:
{
  perSystem = { config, self', pkgs, lib, ... }: {
    devShells.default = pkgs.mkShell {
      name = "learn-claude-code-rs-openai-local-shell";
      inputsFrom = [
        # Note: Removed self'.devShells.rust if it enforces a Nix rust toolchain, 
        # keeping pre-commit hooks intact.
        config.pre-commit.devShell # See ./nix/modules/pre-commit.nix
      ];
      packages = with pkgs; [
        rustup
        just
        nixd # Nix language server
        bacon
        pkg-config
        openssl
      ];

      shellHook = ''
        # Ensure user-level cargo bin (managed by rustup) is in PATH
        export PATH="$HOME/.cargo/bin:$PATH"
        
        # Initialize or verify rustup toolchain from rust-toolchain.toml
        rustup show active-toolchain > /dev/null 2>&1 || rustup show
      '';
    };
  };
}
