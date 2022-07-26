# ELCOR_replacement.dat

## Overview

This file is a direct replacement of the ELCOR.dat file provided in the orignal ellip package of Kennett & Gudmundsson (1996) and contains updated values calculated using EllipticiPy for the ak135 model.

Before using the replacement tables with ellip, the direct access tables must be created as is explained in the README of ellip. If the direct executable already exists then this is trivial.

```
./direct < ELCOR_replacement.dat
```

This command will update elcordir.tbl and ellip will now use the updated tables.

## Citing

If using this file to calculate ellipticity corrections using ellip, please cite the publication corresponding to EllipticiPy. A preprint can be found at: https://arxiv.org/abs/2205.08229. 

More information regarding the construction of ELCOR_replacement.dat can be found in the supplementary materials.

## Further details

The phase names, order and dimensions of ELCOR.dat are hardcoded in the original ellip package and therefore replacement tables must contain the same phases in the same order with the same number of entries per phase in order to be compatible.

One complication of the original ELCOR.dat is that it contains coefficients even where there is not a ray-theoretical arrival and this is highlighted in the ellipcor.help file of the ellip package: *"The presence of a value in the tables does not imply a physical arrival at all distance, depth combinations. Where necessary extrapolation has been used to ensure satisfactory results."* Exactly what is meant by 'necessary extrapolation' is not apparent.

The various assumptions and procedures used to produce ELCOR_replacement.dat are:

-  For points where a phase name has several possible ray paths (e.g. for upper mantle P wave triplications) coefficients for the first arriving ray path are provided.

- Branches of core phases are separated as distinct phases, except for P'P' and S'S' which are explained below.

- For points where we do not expect a ray-theoretical arrival but there is reason to believe that the original values from ELCOR.dat are correct (phases below 180 degrees and good agreement where there are arrivals) we have used the original values.

- For points where we do not expect a ray-theoretical arrival but there is reason to believe that the original values from ELCOR.dat are in doubt, we have linearly extrapolated from the closest calculable values.

- In the original ELCOR.dat, tables for P'P' and S'S' exceed the expected distance ranges for those phases that bottom in the outer core, and in parts of the given range correspond to the df (inner core) branches of the phase. In order to keep ELCOR_replacement.dat compatible, this file must have the same number of entries for each phase as the original ELCOR.dat file. Therefore for P'P' the coefficients given are for P'P'df and for S'S' they are for S'S'ac at the lower end of the distance range and for S'S'df at the upper end of the range. The coefficients for P'P'bc and P'P'df are continuous, as are those for S'S'ac and S'S'df. The only branch not represented by these phase entries is P'P'ab.

- PnS as we understand it (a P wave that descends to and diffracts along the Moho before continuing downwards as an S wave) does not occur totally within the distance range given in the original ELCOR.dat, nor do the coefficients given match those calculated for this phase. As we do not know what ray path these coefficients are for and cannot reproduce them at the exact distances, we omit this phase in ELCOR_replacement.dat. 

## Other remarks

One major disadvantage of using pre-calculated tables is that the exact ray path used to calculate the coefficients is not known to the user. Therefore if in doubt, we encourage all users to use EllipticiPy directly as ObsPy allows easy visualisation of ray paths to confirm that the coefficients are for the intended ray path. 

EllipticiPy also contains a function which produces tabulated coefficients for a given phase and source depth. Coefficients are produced for the full range of ray parameters for which the phase exists, using the same discretisation in slowness as used for the velocity model by ObsPy TauP. Such coefficients can be pre-calculated and then used in a standard interpolation routine.



