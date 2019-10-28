import numpy as np
from alfabet import model

def test_predict():
    results = model.predict(['CC', 'NCCO', 'CF'], verbose=False)

    assert not results[results.molecule == 'CF'].is_valid.any()
    assert results[results.molecule != 'CF'].is_valid.all()

    np.testing.assert_allclose(
        results[results.molecule == 'CC'].bde_pred,
        [90.278282, 99.346191], atol=1E-4)

    np.testing.assert_allclose(
        results[results.molecule == 'NCCO'].bde_pred,
        [89.98849,  82.12242,  98.25096,  99.13476,  92.21609, 92.56299,
         105.120605], atol=1E-4)
