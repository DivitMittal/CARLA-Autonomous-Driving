{
  description = "CARLA-Autonomous-Driving flake";

  outputs = inputs: let
    inherit (inputs.flake-parts.lib) mkFlake;
    specialArgs.customLib = builtins.import (inputs.OS-nixCfg + "/lib/custom.nix") {inherit (inputs.nixpkgs) lib;};
  in
    mkFlake {inherit inputs specialArgs;} ({inputs, ...}: {
      systems = builtins.import inputs.systems;
      imports = [./flake];
    });

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    "nixpkgs-2605".url = "github:NixOS/nixpkgs/nixpkgs-26.05-darwin";
    flake-parts.url = "github:hercules-ci/flake-parts";
    # nix-systems/default dropped x86_64-darwin in c29398b5; pin the last
    # revision that still lists it so lock bumps keep x86_64-darwin outputs.
    systems.url = "github:nix-systems/default/da67096a3b9bf56a91d16901293e51ba5b49a27e";
    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    actions-nix = {
      url = "github:nialov/actions.nix";
      inputs = {
        nixpkgs.follows = "nixpkgs";
        flake-parts.follows = "flake-parts";
        git-hooks.follows = "git-hooks";
      };
    };
    OS-nixCfg = {
      url = "github:DivitMittal/OS-nixCfg";
      flake = false;
    };
  };
}
