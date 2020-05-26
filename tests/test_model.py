import numpy as np
from alfabet import model

def test_predict():
    results = model.predict(['CC', 'NCCO', 'CF', 'B'], verbose=False)

    assert not results[results.molecule == 'B'].is_valid.any()
    assert results[results.molecule != 'B'].is_valid.all()

    np.testing.assert_allclose(
        results[results.molecule == 'CC'].BDE,
        [90.7, 99.8], atol=1., rtol=.05)

    np.testing.assert_allclose(
        results[results.molecule == 'NCCO'].BDE,
        [90.0, 82.1, 98.2, 99.3, 92.1, 92.5, 105.2], atol=1., rtol=.05)

    np.testing.assert_allclose(
        results[results.molecule == 'CF'].BDE,
        [107., 97.5], atol=1., rtol=.05)
