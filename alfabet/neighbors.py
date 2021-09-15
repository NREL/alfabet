import os

import joblib
import tensorflow as tf

from alfabet.drawing import draw_bde
from alfabet.prediction import preprocessor, model, bde_dft

currdir = os.path.dirname(os.path.abspath(__file__))
embedding_model = tf.keras.Model(model.inputs, [model.layers[17].output])

nbrs_pipe = joblib.load(
    os.path.join(currdir, 'model_files/20201012_bond_embedding_nbrs.p.z'))


def pipe_kneighbors(pipe, X):
    Xt = pipe.steps[0][-1].transform(X)
    return pipe.steps[-1][-1].kneighbors(Xt)


def find_neighbor_bonds(smiles, bond_index, draw=True):
    ds = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(item, train=False)
                 for item in (smiles,)),
        output_types=preprocessor.output_types,
        output_shapes=preprocessor.output_shapes).batch(batch_size=1)

    embeddings = embedding_model.predict(ds)
    distances, indices = pipe_kneighbors(nbrs_pipe, embeddings[:, bond_index, :])

    neighbor_df = bde_dft.dropna().iloc[indices.flatten()]
    neighbor_df['distance'] = distances.flatten()
    neighbor_df = neighbor_df.drop_duplicates(
        ['molecule', 'fragment1', 'fragment2']).sort_values('distance')

    if draw:
        neighbor_df['svg'] = neighbor_df.apply(
            lambda x: draw_bde(x.molecule, x.bond_index), 1)

    return neighbor_df
