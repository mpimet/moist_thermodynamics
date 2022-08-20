# moist_thermodynamics

## Description
This repository contains a python module providing constants and functions used for the analysis of moist atmospheric thermodynamic processes.  An effort is made to be as accurate as possible given the assumption of a Rankine-Kirchoff fluid, i.e., zero condensate volume, perfect mixtures of perfect gases, perfect liquids and constancy of specific heats.  

In some cases even more exact treatments will be desired, for instance the more accurate specifications of the saturation vapor pressure are not consistent with the assumptions of a Rankine-Kirchoff fluid, but are useful references.  Those more generally interested in the most accurate treatment are referred to the [IAPWS](http://iapws.org), or the TEOS libraries, e.g., their sea-ice-air library is [available on github](https://github.com/TEOS-10/SIA-Fortran)

The functionality is not meant to be exhaustive, but to provide the basic tools to treate moist thermodynamics in a consistent manner and in ways that allow the tools to be easily incorporated into other programmes, or used as references for simpler more analytically tractable approximations.   

## Usage
Jupyter notebooks are provided in the examples directory to provide illustrative use cases and further information pursuant to choices made in structuring the code, and the functionality it enables.  This includes analyses used in ([Marquet and Stevens (2021)](https://journals.ametsoc.org/view/journals/atsc/79/4/JAS-D-21-0095.1.xml).

## References

Marquet, P., & Stevens, B. (2022). On Moist Potential Temperatures and Their Ability to Characterize Differences in the Properties of Air Parcels, Journal of the Atmospheric Sciences, 79(4), 1089-1103. ([open access pdf version](https://arxiv.org/pdf/2104.01376.pdf)) 


Romps, D.M. (2021), The Rankine–Kirchhoff approximations for moist thermodynamics. QJR Meteorol Soc, 147: 3493-3497. https://doi.org/10.1002/qj.4154

Siebesma, A., Bony, S., Jakob, C., & Stevens, B. (Eds.). (2020). Clouds and Climate: Climate Science's Greatest Challenge. Cambridge: Cambridge University Press. doi:10.1017/9781107447738

## Contributing
Code contributions are welcome.  Please suggest changes on a branch and make a merge request.  Format all changes using Black, and test docstrings using pytest before making merge requests.

## Authors and acknowledgment
The code was written by Bjorn Stevens with contributions from Lukas Kluft and Tobias Kölling. LK and TK also are thanked for expert input in setting up the repository and on how best to structure and code the libraries in ways that enourage intuitive use.  They, also along with Jiawei Bao, Geet George and Hauke Schulz are thanked for their feedback on the thermodynamic analysis.

## License
Copyright 2016-2021 MPI-M, Bjorn Stevens

Code subject to BSD-3-C, SPDX short identifier: BSD-3-Clause, see [license file](LICENSE.md)


## Contact
bjorn.stevens@mpimet.mpg.de

