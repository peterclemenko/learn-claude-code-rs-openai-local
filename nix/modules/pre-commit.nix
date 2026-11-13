{ inputs, ... }:
{
  imports = [
    (inputs.git-hooks + /flake-module.nix)
  ];
  perSystem = { config, self', pkgs, lib, ... }: {
    pre-commit.settings = {
      hooks = {
        nixpkgs-fmt.enable = true;
        rustfmt.enable = true;
        ripsecrets.enable = true;
        trufflehog = {
          enable = true;
          stages = [ "pre-push" ];
          pass_filenames = false;
          entry = "${pkgs.trufflehog}/bin/trufflehog git file://. --since-commit main --results=verified --fail";
        };
      };
    };
  }


