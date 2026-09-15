"""Deterministic image-group partitions; no answer labels influence allocation."""
import random


def image_group(row):
    metadata=row.get('metadata',{})
    if metadata.get('image_id') is not None:
        return 'image:'+str(metadata['image_id'])
    path=row.get('image_path')
    if not path: raise ValueError('Image identity required for partitioning')
    return 'path:'+str(path)


def partition_groups(rows, seed=2026, fit_fraction=.5):
    if not 0 < fit_fraction < 1: raise ValueError('Fraction must be in (0,1)')
    groups=sorted({image_group(x) for x in rows})
    if len(groups)<2: raise ValueError('At least two independent image groups required')
    random.Random(seed).shuffle(groups)
    cut=max(1,min(len(groups)-1,int(len(groups)*fit_fraction)))
    fit=set(groups[:cut])
    return ([x for x in rows if image_group(x) in fit],
            [x for x in rows if image_group(x) not in fit])
