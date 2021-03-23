# daart testing

### Unit testing

daart has partial unit testing coverage. The package uses the `pytest` package for unit testing. 
This package must be installed separately (in the daart conda environment) and can be run from the 
command line within the top-level `daart` directory:

```bash
(daart) $: pytest
```

where `(daart)` indicates the shell is in the `daart` conda environment.

As of March 2021 most helper functions have been unit tested, though modeling code (i.e. `pytorch` 
code) has not.

### Integration testing

daart also has a rudimentary integration test. From the top-level `daart` directory, run the 
following from the command line:

```bash
(daart) $: python tests/integration.py
```

The integration test will 

1. create temporary data/results directories
2. create simulated data
3. fit the following models:
    * temporal mlp
4. delete the temporary data/directories

The integration test checks that all models finished training. 
Models are only fit for a single epoch with a small amount of data, so total fit time should be 
less than one minute. 
The purpose of the integration test is to ensure the `pytorch` models are fitting properly, and 
that all path handling functions are working.
