"""Package marker.

It makes ``tests`` a regular package, so ``offline_stage.py`` imports
``tests.test_offline_stage`` from this directory and not from another ``tests``
that happens to sit earlier on the path.
"""
