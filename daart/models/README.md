A checklist for adding a new model to the daart package:
===

Model-related code
---

* define a new class in `daart.models` package
    * update the `daart.models.base.Segmenter.build_model()` method to include the new model
* required function updates:
    * `daart.io.get_model_params` [UPDATE UNIT TEST!]
* update relevant configs (e.g. add any new hyperparameters, but try to resuse old ones if possible)


Testing
---

* add new model to integration script `tests/integration.py`
    * add to `MODELS_TO_FIT` list at top of file 
    * update `define_new_config_values()`
*  run tests
    * unit tests: from daart parent directory run `pytest`
    * integration test: from daart parent directory run `python tests/integration.py`


Documentation
---

* complete all docstrings in new functions
