"""Dataset + PifPaf mask resolution for the part-prompt scripts (processor/train_part_prompts_stage{1,2}.py).

One rule for masks, three datasets: the mask of `<dataset_dir>/<rel>.jpg` is `<dataset_dir>/masks/<variant>/<rel>.npy`
- the mask tree mirrors the image tree below the directory that holds the split folders. That is exactly the layout
of BPBreID's released Market-1501 masks (`masks/pifpaf_maskrcnn_filtering/bounding_box_train/<stem>.npy`) and of
`reid_masks/compute_masks.py`'s MSMT17 output (`masks/pifpaf_maskrcnn_filtering/mask_train_v2/<pid>/<stem>.npy`),
so nothing has to special-case a dataset and DukeMTMC-reID follows for free once its masks are generated.

The repo's own dataset classes are left untouched (train_clipreid.py still uses them); the two subclasses here only
fix directory names that do not match the copies on disk:
  * MSMT17_V2 ships `mask_train_v2` / `mask_test_v2` instead of V1's `train` / `test` (faces are blurred in V2).
  * DukeMTMC-reID sits in `dukemtmc-reid/`, not the `dukemtmc/` the upstream class assumes.
Both resolve through DIR_CANDIDATES, so a differently named copy is found without editing this file.

Masks are only ever needed for the *train* split (stage 1 pools with them, stage 2 supervises the LPIM attention
with them); query/gallery are read as plain images at evaluation. `resolve_masks` therefore reports on the train
split alone, and returns the expected path even when it does not exist yet - that path is what
`reid_masks/compute_masks.py --dataset <dataset_dir>` fills in.
"""
import os
import os.path as osp

from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmcreid import DukeMTMCreID

MASKS_VARIANT = 'pifpaf_maskrcnn_filtering'        # BPBreID masks_dir name; 'pifpaf' is the unfiltered variant
MASK_SUFFIX = {                                    # bpbreid masks_dirs: (36 channels, no background, suffix)
    'pifpaf_maskrcnn_filtering': '.npy',           # PifPaf fields x the Mask R-CNN person silhouette
    'pifpaf': '.jpg.confidence_fields.npy',        # raw PifPaf fields, no person filtering
}


class MSMT17V2(MSMT17):
    """MSMT17 as distributed in the V2 archive: same list files, `mask_{train,test}_v2` image folders."""
    dataset_dir = 'MSMT17_V2'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'mask_test_v2')
        for split in ['train', 'val', 'query', 'gallery']:
            setattr(self, f'list_{split}_path', osp.join(self.dataset_dir, f'list_{split}.txt'))
        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        train += self._process_dir(self.train_dir, self.list_val_path)
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print('=> MSMT17 (V2) loaded')
            self.print_dataset_statistics(train, query, gallery)
        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(gallery)


class DukeMTMCreIDPath(DukeMTMCreID):
    """DukeMTMC-reID with the directory it actually sits in (upstream hardcodes `dukemtmc`)."""
    dataset_dir = 'dukemtmc-reid'


# name -> (class, candidate parent dirs under DATA_ROOT, sub-path holding the split folders, train split folder)
DATASETS = {
    'market1501': dict(cls=Market1501, dirs=['market1501', 'Market-1501-v15.09.15'], inner='', train='bounding_box_train'),
    'msmt17': dict(cls=MSMT17V2, dirs=['MSMT17_V2'], inner='', train='mask_train_v2'),
    'dukemtmc': dict(cls=DukeMTMCreIDPath, dirs=['dukemtmc-reid', 'dukemtmc'], inner='DukeMTMC-reID', train='bounding_box_train'),
}


def resolve_dir(name, root):
    """The directory the copy on disk actually uses, picked from DIR candidates."""
    for candidate in DATASETS[name]['dirs']:
        if osp.isdir(osp.join(root, candidate)):
            return candidate
    raise FileNotFoundError(f"{name}: none of {DATASETS[name]['dirs']} found under {root}")


def build_dataset(name, root, verbose=True):
    """Returns (dataset, dataset_dir) where dataset_dir is the folder that directly holds the split folders."""
    assert name in DATASETS, f'dataset must be one of {list(DATASETS)}'
    spec, found = DATASETS[name], resolve_dir(name, root)
    cls = type(spec['cls'].__name__, (spec['cls'],), {'dataset_dir': found})    # no mutation of the shared class
    dataset = cls(root=root, verbose=verbose)
    return dataset, osp.normpath(osp.join(root, found, spec['inner']))


def masks_root(dataset_dir, variant=MASKS_VARIANT):
    """<dataset_dir>/masks/<variant> - the BPBreID layout, and what reid_masks/compute_masks.py writes."""
    return osp.join(dataset_dir, 'masks', variant)


def mask_path(img_path, dataset_dir, masks=None, variant=MASKS_VARIANT):
    """Mask of one image: its path relative to dataset_dir, under the mask root, with the variant's suffix
    (bpbreid infer_masks_path: the image extension is dropped and the suffix appended, so the unfiltered
    variant's name carries its own '.jpg')."""
    rel = osp.relpath(osp.abspath(img_path), osp.abspath(dataset_dir))
    return osp.join(masks or masks_root(dataset_dir, variant), osp.splitext(rel)[0] + MASK_SUFFIX[variant])


def resolve_masks(name, dataset, dataset_dir, logger=print, require=True, paths=None,
                  variant=MASKS_VARIANT, masks=None):
    """Log the resolved mask directory and how complete it is; returns it whether or not it exists.

    `masks` overrides the location for a mask set kept outside the dataset directory (a downloaded release on
    another disk, say); `variant` picks which BPBreID mask set to read. `paths` limits the completeness check to
    the images that will actually be read (stage 1 with --num-ids uses a few identities, and those may be the only
    ones generated so far); the default is the whole train split. Missing masks are not fixed here: the variable
    points at the expected location, which is what `reid_masks/compute_masks.py --dataset <dataset_dir>` fills.
    """
    masks = masks or masks_root(dataset_dir, variant)
    wanted = list(paths) if paths is not None else [item[0] for item in dataset.train]
    missing = [p for p in wanted if not osp.exists(mask_path(p, dataset_dir, masks, variant))]
    logger(f'{name}: images {dataset_dir}, masks {masks} ({len(wanted) - len(missing)}/{len(wanted)} present)')
    if missing:
        message = (f'{name}: {len(missing)} masks missing under {osp.join(masks, DATASETS[name]["train"])}, '
                   f'e.g. {mask_path(missing[0], dataset_dir, masks, variant)} - generate them with '
                   f'`python compute_masks.py --dataset {dataset_dir} --splits {DATASETS[name]["train"]}` '
                   f'in ../reid_masks, or point --masks-dir at a pre-saved set')
        if require:
            raise FileNotFoundError(message)
        logger(message)
    return masks
