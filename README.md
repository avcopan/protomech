# ProtoMech

Prototypes of new submodules for the AutoMech code.

## Installation

> [!NOTE]
> This code has been set up to run on Linux or Mac OS.
> If you are on a Windows machine, you can install
> [WSL](https://learn.microsoft.com/en-us/windows/wsl/setup/environment#get-started)
> and run the code from your WSL terminal.

To get set up, first install the Pixi package manager by following [these instructions](https://pixi.sh/latest/#installation).
Then you can install this code as follows:
```
git clone https://github.com/avcopan/protomech.git
cd protomech
pixi install -e all
```

## Usage

### RMG

*Example.* This code can be used to analyze data in an RMG input mechanism.
To see how it works, you can navigate to the RMG example and open up the Jupyter
notebook as follows.
```
cd examples/rmg/
pixi run -e all jupyter lab visualize.ipynb
```
If this doesn't automatically open a Jupyter window in your browser, you may
need to `Ctrl + Click` the URL that gets printed by Jupyter lab.
You can run the code there by clicking the ⏩ icon to "Restart the kernel and run all cells."

*Explanation.* You can import an RMG mechanism as follows, where you will insert the names of the ChemKin and species adjacency list files, which must be in the same directory as the notebook.
```
import automech

mech = automech.io.rmg.read.mechanism(
    rxn_inp="<chemkin reaction file>",
    spc_inp="<species adjacency list file>",
)
```
You can then use the `automech.display_species` to display species.
```
automech.display_species(mech)  # <-- display all species
automech.display_species(       # <-- display species CO2(13) and C5H7O(2)
    mech,
    spc_vals_=["CO2(13)", "C5H7O(2)"],
)
```
You can then use the `automech.display_reactions` to display reactions.
```
automech.display_reactions(mech)  # <-- display all reactions
automech.display_reactions(       # <-- display specific reactions by equation
    mech,
    eqs=[
        "HOCO(17) = CO(12) + OH(4)",
        "C5H7O(1) = C5H7O(2)",
    ],
)
```
