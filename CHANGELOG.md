# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.0] - 2023-08-23

### Changed

- Major refactoring and revamping of the functionality from Verion 0 to 1
- Refactored pipeline for computing cellular niches
- Ability to run more exact kernel density estimation in addition to the grid approximation
- New visualization tools for exploring cell type enrichment in cellular niches
- New visualization tools for visualizing ligand-receptor co-expression in neighboring cells
- Ability to download ligand-receptor pairs from OmniPath
- Removal of many functions deemed reduntant
- New tutorial notebook

## [0.6.0] - 2022-12-16

### Changed

-   Changed `cell_contact_embedding_zoom` to `embedding_zoom` and abstracted functionality

### Removed

-   Removed volcano plot function

## [0.5.0] - 2022-12-13

### Added

-   Added ligand-receptor interaction calculation, statistical, and plotting functions
-   Added cell contact zoom plot for more finely focused examination of cell contacts
-   Added embedding filter plot for highlighting only certain groups while coloring by another metric

### Changed

-   Renamed arguments to `mb.calc.neighborhood_profile` for better understanding

## [0.4.4] - 2022-11-29

### Added

-   `mb.calc.neighborhood_profile` can take in a number of nearest neighbors to consider instead of radius

### Changed

-   `mb.plot.cell_contact_heatmap` sorts categories

## [0.4.3] - 2022-11-28

### Changed

-   `mb.calc.shortest_distances` now returns a DataFrame for easier indexing
-   `mb.plot.cell_transcript_proximity` uses predetermined colors if previously added by Scanpy

## [0.4.2] - 2022-11-23

### Fixed

-   Fixed calculation in shortest distance statistical test

## [0.4.1] - 2022-11-22

### Added

-   Basis customization to functions using coordinates

## [0.4.0] - 2022-11-21

### Added

-   This changelog
-   Neighborhood profile function

### Fixed

-   Plotting function legends now better located
-   `load_merscope` maintains transcript ids

## [0.3.5] - 2022-11-21

### Added

-   Enhancements to plotting functions with better labels/legends

### Fixed

-   Shortest distances permutation test now includes pseudocount

### Removed

-   Removed unnecessary columns from `detected_transcripts.csv` when using `load_merscope`

## [0.3.4] - 2022-11-14

### Added

-   Tutorial notebook added to documentation

### Changed

-   Kernel density with multiple groups are scaled individually

### Fixed

-   Kernel density plot bug with multiple displays
-   Docstring return descriptions

## [0.3.3] - 2022-11-11

### Fixed

-   Address issue from v0.3.2 (now yanked)

## [0.3.1] - 2022-11-09

### Added

-   Cell contact counting function

### Changed

-   Default `show = True` for plotting functions for consistency with scverse ecosystem

## [0.3.0] - 2022-11-08

### Added

-   FOV column added to `adata.obs` when using `load_merscope`

### Changed

-   Plotting cell contact embedding now requires a `group` argument

### Fixed

-   Various bug fixes in cell contact, cell-transcript proximity, and shortest distances

## [0.2.2] - 2022-11-03

### Added

-   FOV column added to `adata.obs` when using `load_merscope`

## [0.2.1] - 2022-11-02

### Fixed

-   Updated kernel density documentation

## [0.2.0] - 2022-11-01

### Added

-   Integration with cookiecutter-scverse

## [0.1.7] - 2022-10-31

### Added

-   Option to subset by coordinates in `subset_cells`
-   Better documentation for cell contact calculation
-   Add basis argument to cell contact embedding plot

## [0.1.6] - 2022-10-28

### Changed

-   Cell contact count removes duplicate contacts from the dictionary

## [0.1.5] - 2022-10-26

### Changed

-   Cell contact statistical method no longer uses z-test

## [0.1.4] - 2022-10-25

### Fixed

-   Cell contact counts returning 0 due to inaccurate comparison

## [0.1.3] - 2022-10-25

### Changed

-   Cell contact statistical method only returns p-values if `split_groups = True`

## [0.1.2] - 2022-10-25

### Changed

-   Cell contact statistical method uses updated counting method

## [0.1.1] - 2022-10-25

### Fixed

-   Cell contact statistical test array initialization bug

## [0.1.0] - 2022-10-25

### Added

-   Volcano plots
-   Cell contact embedding, heatmap, and histplot separated functions

### Removed

-   Cell contact unified plotting function

## [0.0.3] - 2022-10-25

### Changed

-   `load_merscope` can use a cached object in a few different ways

## [0.0.2] - 2022-09-06

### Added

-   Utility function for loading an AnnData object from MERSCOPE data
-   Functions for investigating cell-transcript proximity

## [0.0.1] - 2022-08-25

### Added

-   Initial functions for cell contact, kernel density, and shortest distances
-   Utility function for subsetting cells
