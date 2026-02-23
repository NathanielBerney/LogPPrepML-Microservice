import os
import numpy as np
import joblib
import tensorflow as tf
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec

# Suppress TF logs for a cleaner CLI experience
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class LogPMLHandler:
    AVAILABLE_PROPERTIES = ["ML_LogP_NN", "ML_LogP_Ridge"]

    def __init__(self, model_dir: str = "./logp_pred_ml"):
        self.models = {}
        self.w2vec = None
        self._load_assets(model_dir)

    def _load_assets(self, path: str) -> None:
        # 1. Load the Mol2Vec "Translator"
        self.w2vec = word2vec.Word2Vec.load(os.path.join(path, 'model_300dim.pkl'))
        
        # 2. Load the Neural Network (Keras)
        # Disable GPU for consistency with your notebook setup
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        self.models["ML_LogP_NN"] = tf.keras.models.load_model(os.path.join(path, 'logP_model.keras'))
        
        # 3. Load the Classical Pipeline (SVR/Ridge)
        self.models["ML_LogP_Ridge"] = joblib.load(os.path.join(path, 'best_pipeline.pkl'))

    def _featurize(self, mol) -> np.ndarray:
        """Converts RDKit Mol to the 113-dim hybrid vector."""
        # A. Mol2Vec (100 dimensions usually, or 300 depending on your notebook setup)
        sentence = MolSentence(mol2alt_sentence(mol, radius=1))
        
        # Internal helper to handle the vector summation
        keys = set(self.w2vec.wv.key_to_index.keys())
        this_vec = [self.w2vec.wv[word] for word in sentence if word in keys]
        
        if not this_vec:
            mol2vec_part = np.zeros(self.w2vec.vector_size)
        else:
            mol2vec_part = np.sum(this_vec, axis=0)
        
        # B. RDKit Descriptors (The 13 specific ones from your notebook)
        rdkit_part = [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol), Descriptors.NumValenceElectrons(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumAliphaticRings(mol),
            Descriptors.HeavyAtomCount(mol), Descriptors.RingCount(mol),
            Descriptors.FractionCSP3(mol), Descriptors.MolMR(mol)
        ]
        
        # Combine into a single row
        full_features = np.concatenate([mol2vec_part, rdkit_part])
        return full_features.reshape(1, -1) # Reshape for a single prediction

    def process_multiple_properties(self, smi: str, property_list: List[str]) -> Dict[str, Any]:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES")

            features = self._featurize(mol)
            results = {}

            for prop in property_list:
                if prop == "ML_LogP_NN":
                    pred = self.models["ML_LogP_NN"].predict(features, verbose=0)
                    results[prop] = float(pred.flatten()[0])
                
                elif prop == "ML_LogP_Ridge":
                    # Scikit-learn models handle 2D arrays directly
                    pred = self.models["ML_LogP_Ridge"].predict(features)
                    results[prop] = float(pred[0])

            return {
                "smiles": smi,
                "status": "success",
                "results": results,
                "error": None
            }

        except Exception as e:
            return {"smiles": smi, "status": "error", "results": {}, "error": str(e)}

    def batch_predict(self, smiles_list: List[str], props: List[str]) -> List[Dict]:
        return [self.process_multiple_properties(s, props) for s in smiles_list]

if __name__ == "__main__":
    # Ensure your .pkl and .keras files are in the same folder as this script
    handler = LogPMLHandler()
    res = handler.process_multiple_properties("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", ["ML_LogP_NN", "ML_LogP_Ridge"])
    print(res)