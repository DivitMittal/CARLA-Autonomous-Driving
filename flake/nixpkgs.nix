{inputs, ...}: {
  perSystem = {system, ...}: {
    # nixpkgs-unstable (26.11+) no longer evaluates on Intel Macs, so pin them
    # to the last nixpkgs branch that still supports x86_64-darwin. Other
    # systems keep flake-parts' default (inputs.nixpkgs).
    _module.args.pkgs =
      if system == "x86_64-darwin"
      then inputs."nixpkgs-2605".legacyPackages.${system}
      else inputs.nixpkgs.legacyPackages.${system};
  };
}
