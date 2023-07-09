from dataclasses import dataclass
from pathlib import Path


@dataclass
class Asset:
    path: Path
    url: str
    hash: str
    group: str


ASSETS = [
    # BSR-GAN
    Asset(path=Path('assets/bsrganx2_160x160.xml'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2Fbsrganx2_160x160.xml',
          hash='21b172609694d4094ebe86a4ce6902df2cde405e',
          group='BSR-GAN'),
    Asset(path=Path('assets/bsrganx2_160x160.bin'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2Fbsrganx2_160x160.bin',
          hash='d543729c721b29488fd50e80665bd9f8efd514aa',
          group='BSR-GAN'),

    # Face Detection
    # Maybe use this instead?
    # 'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.3/models_bin/1/face-detection-adas-0001/FP32/'
    Asset(path=Path('assets/face-detection-adas-0001.xml'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2Fface-detection-adas-0001.xml',
          hash='dbba0527caf53d50367b07d8ada5b030fdb6a699',
          group='Face-Detection'),
    Asset(path=Path('assets/face-detection-adas-0001.bin'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2Fface-detection-adas-0001.bin',
          hash='79ca338fff14ddb54a3763c8b3d3f6a0b11bedaf',
          group='Face-Detection'),

    # CFRGAN
    Asset(path=Path('assets/CFRNet_G_ep55_vgg.pth'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2FCFRNet_G_ep55_vgg.pth',
          hash='3a81c293623b122bbc51dea140565659c5078015',
          group='CFRGAN'),
    Asset(path=Path('assets/trained_weights_occ_3d.pth'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2Ftrained_weights_occ_3d.pth',
          hash='bc1990a3ff701585503615c83982474e2798e62a',
          group='CFRGAN'),
    Asset(path=Path('assets/mmRegressor/BFM/BFM_model_80.mat'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2FmmRegressor%2FBFM%2FBFM_model_80.mat',
          hash='4d474a3ee99073986a0bb1ddb36d1b0a1e306d8b',
          group='CFRGAN'),
    Asset(path=Path('assets/mmRegressor/BFM/similarity_Lm3D_all.mat'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2FmmRegressor%2FBFM%2Fsimilarity_Lm3D_all.mat',
          hash='863be6cce68aae53868af4a3f725845b812915ec',
          group='CFRGAN'),
    Asset(path=Path('assets/mmRegressor/network/th_model_params.pth'),
          url='https://objectstorage.us-chicago-1.oraclecloud.com/n/axw9w7h9hwka/b/atfal-ai/o/assets%2FmmRegressor%2Fnetwork%2Fth_model_params.pth',
          hash='de971d5646c060c4faa314a8962a0e5c9b5a9d3a',
          group='CFRGAN'),

]
