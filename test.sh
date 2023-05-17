#!/bin/bash

# tests/test_dfp.py::test_dfp_roleg

rm -f core && \
    rm -f ~/.cache/morpheus/log/morpheus.log && \
    pytest -x --run_slow --log_level=DEBUG \
        --log-file /home/dagardner/.cache/morpheus/log/pytest.log \
        -l -s 
