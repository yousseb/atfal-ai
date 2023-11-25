import os
from pathlib import Path

from md5checker import make_hash
from downloer_base import DownloaderBase
from asset_config import Asset, ASSETS


class AssetDownloader(DownloaderBase):
    def __init__(self):
        super(AssetDownloader, self).__init__()

    def verify_asset(self, asset: Asset) -> bool:
        if not asset.path.exists():
            print(f'File not found for asset: {asset.path} in group: {asset.group}')
            return False
        calculated_hash = make_hash(asset.path.absolute(), algo='sha1')
        if calculated_hash.lower() != asset.hash.lower():
            print(f'Hashes not matching for: {asset.path} in group:  {asset.group} - '
                  f'wanted: {asset.hash} - calculated: {calculated_hash}')
            return False
        return True

    def verify_assets(self) -> bool:
        assets_ok = True
        for asset in ASSETS:
            if not self.verify_asset(asset):
                assets_ok = False
        return assets_ok

    def delete_asset(self, asset: Asset) -> None:
        asset.path.unlink(missing_ok=True)

    def download_asset(self, asset: Asset) -> bool:
        print(f'Downloading asset: {asset.path} in group: {asset.group}')
        Path(asset.path.parent).mkdir(parents=True, exist_ok=True)
        return self.download_file(asset.url, asset.path)

    def ensure_asset(self, asset: Asset) -> bool:
        if not self.verify_asset(asset):
            if asset.path.exists():
                self.delete_asset(asset)
            self.download_asset(asset)
        return self.verify_asset(asset)

    def ensure_assets(self) -> bool:
        assets_ok = True
        for asset in ASSETS:
            if not self.ensure_asset(asset):
                assets_ok = False
        return assets_ok


if __name__ == '__main__':
    cwd = Path(os.getcwd())
    if (list(cwd.parts)[-1:][0]).lower() == 'common':
        os.chdir(Path('../..').absolute())
    AssetDownloader().ensure_assets()
