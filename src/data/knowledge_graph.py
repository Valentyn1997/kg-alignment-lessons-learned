# coding=utf-8
"""
Loading utilities for knowledge graphs.
"""
import logging
import os
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from os import path
import re
from typing import Collection, List, Mapping, Optional, Set, TextIO, Tuple, Union

import numpy
import requests
import torch

from utils.common import download_file_from_google_drive

DATASETS = {'dbp15k', 'dwy100k', 'wk3l', 'dbp15k_jape', 'dwy100k'}


class Extractor:
    def extract(self, archive_path: str, cache_root: Optional[str] = None):
        if cache_root is None:
            cache_root = path.dirname(archive_path)
        self._extract(archive_path=archive_path, cache_root=cache_root)
        logging.info(f'Extracted {archive_path} to {cache_root}.')

    def _extract(self, archive_path: str, cache_root: str):
        raise NotImplementedError


class ZipExtractor(Extractor):
    def _extract(self, archive_path: str, cache_root: str):
        # self.fix_bad_zip_file(archive_path)
        with zipfile.ZipFile(file=archive_path) as zf:
            zf.extractall(path=cache_root)

    # @staticmethod
    # def fix_bad_zip_file(zipFile):
    #     f = open(zipFile, 'r+b')
    #     data = f.read()
    #     pos = data.find(b'\x50\x4b\x05\x06')  # End of central directory signature
    #     if (pos > 0):
    #         logging.info("Trancating file at location " + str(pos + 22) + ".")
    #         f.seek(pos + 22)  # size of 'ZIP end of central directory record'
    #         f.truncate()
    #         f.close()


class TarExtractor(Extractor):
    def _extract(self, archive_path: str, cache_root: str):
        with open(archive_path, 'rb') as archive_file:
            with tarfile.open(fileobj=archive_file) as tf:
                tf.extractall(path=cache_root)


def add_self_loops(
    triples: torch.LongTensor,
    relation_label_to_id: Mapping[str, int],
    self_loop_relation_name: str = 'self_loop',
) -> Tuple[torch.LongTensor, Mapping[str, int]]:
    """Add self loops with dummy relation.

    For each entity e, add (e, self_loop, e).

    :param triples: shape: (n, 3)
    :param relation_label_to_id:
    :param self_loop_relation_name: the name of the self-loop relation. Must not exist.

    :return: cat(triples, self_loop_triples)
    """
    s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]

    # check if name clashes might occur
    if self_loop_relation_name in relation_label_to_id.keys():
        raise AssertionError(f'There exists a relation "{self_loop_relation_name}".')

    # Append inverse relations to translation table
    num_relations = len(relation_label_to_id)
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    updated_relation_label_to_id.update({self_loop_relation_name: num_relations + 1})
    assert len(updated_relation_label_to_id) == num_relations + 1

    # create self-loops triples
    assert (p < num_relations).all()
    e = torch.tensor(sorted(set(map(int, s)).union(map(int, o))), dtype=torch.long)
    p_self_loop = num_relations + 1
    p_self_loop = torch.ones_like(e) * p_self_loop
    self_loop_triples = torch.stack([e, p_self_loop, e], dim=1)

    all_triples: torch.LongTensor = torch.cat([triples, self_loop_triples], dim=0)

    return all_triples, updated_relation_label_to_id


def add_inverse_triples(
    triples: torch.LongTensor,
    relation_label_to_id: Mapping[str, int],
) -> Tuple[torch.LongTensor, Mapping[str, int]]:
    """Create an append inverse triples.

    For each triple (s, p, o), an inverse triple (o, p_inv, s) is added.

    :param triples: shape: (n, 3)
    :param relation_label_to_id:

    :return: cat(triples, inverse_triples)
    """
    s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]

    # check if name clashes might occur
    suspicious_relations = sorted(k for k in relation_label_to_id.keys() if k.endswith('_inv'))
    if len(suspicious_relations) > 0:
        raise AssertionError(f'Some of the inverse relations did already exist! Suspicious relations: {suspicious_relations}')

    # Append inverse relations to translation table
    num_relations = len(relation_label_to_id)
    updated_relation_label_to_id = {r_label: r_id for r_label, r_id in relation_label_to_id.items()}
    updated_relation_label_to_id.update({r_label + '_inv': r_id + num_relations for r_label, r_id in relation_label_to_id.items()})
    assert len(updated_relation_label_to_id) == 2 * num_relations

    # create inverse triples
    assert (p < num_relations).all()
    p_inv = p + num_relations
    inverse_triples = torch.stack([o, p_inv, s], dim=1)

    all_triples: torch.LongTensor = torch.cat([triples, inverse_triples], dim=0)

    return all_triples, updated_relation_label_to_id


@dataclass
class KnowledgeGraph:
    """A knowledge graph."""
    #: The triples, shape: (n, 3)
    triples: torch.LongTensor

    #: The mapping from entity labels to IDs
    entity_label_to_id: Optional[Mapping[str, int]]

    #: The mapping from relations labels to IDs
    relation_label_to_id: Optional[Mapping[str, int]]

    #: Whether inverse triples have been added
    inverse_triples: bool = False

    #: Whether self-loops have been added.
    self_loops: bool = False

    @property
    def num_triples(self) -> int:
        return self.triples.shape[0]

    @property
    def num_entities(self) -> int:
        return len(self.entity_label_to_id)

    @property
    def num_relations(self) -> int:
        return len(self.relation_label_to_id)

    def __str__(self):
        return f'{self.__class__.__name__}(num_triples={self.num_triples}, num_entities={self.num_entities}, num_relations={self.num_relations}, inverse_triples={self.inverse_triples}, self_loops={self.self_loops})'


@dataclass
class KGAlignment:
    """An alignment between two knowledge graphs."""
    #: The entity alignment used for training.
    entity_alignment_train: torch.LongTensor

    #: The entity alignment used for testing.
    entity_alignment_test: torch.LongTensor

    #: The relation alignment used for training.
    relation_alignment_train: Optional[torch.LongTensor]

    #: The relation alignment used for testing.
    relation_alignment_test: Optional[torch.LongTensor]

    @property
    def num_train_alignments(self) -> int:
        return self.entity_alignment_train.shape[1]

    @property
    def num_test_alignments(self) -> int:
        return self.entity_alignment_test.shape[1]

    def __str__(self):
        return f'{self.__class__.__name__}(num_ea_train={self.num_train_alignments}, num_ea_test={self.num_test_alignments})'


def _apply_compaction(
    triples: Optional[torch.LongTensor],
    compaction: Mapping[int, int],
    columns: Union[int, Collection[int]],
    dim: int = 0,
) -> Optional[torch.LongTensor]:
    if compaction is None or triples is None:
        return triples
    if isinstance(columns, int):
        columns = [columns]
    if dim not in {0, 1}:
        raise KeyError(dim)
    triple_shape = triples.shape
    if dim == 1:
        triples = triples.t()
    new_cols = []
    for c in range(triples.shape[1]):
        this_column = triples[:, c]
        if c in columns:
            new_cols.append(torch.tensor([compaction[int(e)] for e in this_column]))
        else:
            new_cols.append(this_column)
    new_triples = torch.stack(new_cols, dim=1 - dim)
    assert new_triples.shape == triple_shape
    return new_triples


def _compact_columns(
    triples: torch.LongTensor,
    label_to_id_mapping: Mapping[str, int],
    columns: Union[int, Collection[int]],
    dim=0,
) -> Tuple[torch.LongTensor, Optional[Mapping[str, int]], Optional[Mapping[int, int]]]:
    ids = label_to_id_mapping.values()
    num_ids = len(ids)
    assert len(set(ids)) == len(ids)
    max_id = max(ids)
    if num_ids < max_id + 1:
        compaction = dict((old, new) for new, old in enumerate(sorted(ids)))
        assert set(compaction.keys()) == set(label_to_id_mapping.values())
        assert set(compaction.values()) == set(range(num_ids))
        new_triples = _apply_compaction(triples, compaction, columns, dim=dim)
        new_mapping = {label: compaction[_id] for label, _id in label_to_id_mapping.items()}
        logging.info(f'Compacted: {max_id} -> {num_ids - 1}')
    else:
        compaction = None
        new_triples = triples
        new_mapping = label_to_id_mapping
        logging.info(f'No compaction necessary.')
    return new_triples, new_mapping, compaction


def _compact_graph(graph: KnowledgeGraph) -> Tuple[KnowledgeGraph, Optional[Mapping[int, int]], Optional[Mapping[int, int]]]:
    if graph.inverse_triples:
        raise NotImplementedError

    triples0 = graph.triples

    # Compact entities
    triples1, compact_entity_label_to_id, entity_compaction = _compact_columns(triples=triples0, label_to_id_mapping=graph.entity_label_to_id, columns=(0, 2))

    # Compact relations
    triples2, compact_relation_label_to_id, relation_compaction = _compact_columns(triples=triples1, label_to_id_mapping=graph.relation_label_to_id, columns=(1,))

    # Compile to new knowledge graph
    compact_graph = KnowledgeGraph(triples=triples2, entity_label_to_id=compact_entity_label_to_id, relation_label_to_id=compact_relation_label_to_id)

    return compact_graph, entity_compaction, relation_compaction


def _compact_single_alignment(
    single_alignment: torch.LongTensor,
    match_compaction: Mapping[int, int],
    ref_compaction: Mapping[int, int],
) -> torch.LongTensor:
    compact_single_alignment = single_alignment
    for col, compaction in enumerate([match_compaction, ref_compaction]):
        compact_single_alignment = _apply_compaction(triples=compact_single_alignment, compaction=compaction, columns=col, dim=1)
    return compact_single_alignment


def _compact_knowledge_graph_alignment(
    alignment: KGAlignment,
    match_entity_compaction: Mapping[int, int],
    ref_entity_compaction: Mapping[int, int],
    match_relation_compaction: Optional[Mapping[int, int]] = None,
    ref_relation_compaction: Optional[Mapping[int, int]] = None,
) -> KGAlignment:
    # Entity compaction
    compact_entity_alignment_train = _compact_single_alignment(single_alignment=alignment.entity_alignment_train, match_compaction=match_entity_compaction, ref_compaction=ref_entity_compaction)
    compact_entity_alignment_test = _compact_single_alignment(single_alignment=alignment.entity_alignment_test, match_compaction=match_entity_compaction, ref_compaction=ref_entity_compaction)

    # Relation compaction
    compact_relation_alignment_train = _compact_single_alignment(single_alignment=alignment.relation_alignment_train, match_compaction=match_relation_compaction, ref_compaction=ref_relation_compaction)
    compact_relation_alignment_test = _compact_single_alignment(single_alignment=alignment.relation_alignment_test, match_compaction=match_relation_compaction, ref_compaction=ref_relation_compaction)

    return KGAlignment(
        entity_alignment_train=compact_entity_alignment_train,
        entity_alignment_test=compact_entity_alignment_test,
        relation_alignment_train=compact_relation_alignment_train,
        relation_alignment_test=compact_relation_alignment_test,
    )


def _compact(
    match_graph: KnowledgeGraph,
    ref_graph: KnowledgeGraph,
    alignment: KGAlignment,
) -> Tuple[KnowledgeGraph, KnowledgeGraph, KGAlignment]:
    compact_match_graph, match_entity_compaction, match_relation_compaction = _compact_graph(graph=match_graph)
    compact_ref_graph, ref_entity_compaction, ref_relation_compaction = _compact_graph(graph=ref_graph)
    compact_alignment = _compact_knowledge_graph_alignment(
        alignment=alignment,
        match_entity_compaction=match_entity_compaction,
        ref_entity_compaction=ref_entity_compaction,
        match_relation_compaction=match_relation_compaction,
        ref_relation_compaction=ref_relation_compaction,
    )
    return compact_match_graph, compact_ref_graph, compact_alignment


class KnowledgeGraphAlignmentDataset:
    """Contains a lazy reference to a knowledge graph alignment data set."""

    #: The URL where the data can be downloaded from
    url: str

    #: The directory where the datasets will be extracted to
    cache_root: str

    #: The first knowledge graph
    _match_graph: KnowledgeGraph

    #: The second knowledge graph
    _ref_graph: KnowledgeGraph

    #: The alignment
    _alignment: KGAlignment

    def __init__(
        self,
        url: str,
        cache_root: Optional[str] = None,
        inverse_triples: bool = False,
        self_loops: bool = False,
        archive_file_name: Optional[str] = None,
        extractor: Extractor = TarExtractor(),
        **kwargs
    ) -> None:
        """Initialize the data set."""
        self.inverse_triples = inverse_triples
        self.self_loops = self_loops
        if archive_file_name is None:
            archive_file_name = url.rsplit(sep='/', maxsplit=1)[-1]
        self.archive_file_name = archive_file_name

        if cache_root is None:
            cache_root = tempfile.gettempdir()
        self.cache_root = path.join(cache_root, self.__class__.__name__.lower())
        self.url = url

        os.makedirs(self.cache_root, exist_ok=True)

        # Check if files already exist
        archive_path = path.join(self.cache_root, self.archive_file_name)
        if not path.isfile(archive_path):
            if 'drive.google.com' in self.url:
                _id = self.url.split('?id=')[1]
                download_file_from_google_drive(id=_id, destination=archive_path)
            else:
                logging.info(f'Requesting dataset from {self.url}')
                r = requests.get(url=self.url)
                assert r.status_code == requests.codes.ok
                with open(archive_path, 'wb') as archive_file:
                    archive_file.write(r.content)

        else:
            logging.info(f'Skipping to download from {self.url} due to existing files in {self.cache_root}.')

        # Extract files
        # TODO: Could be skipped, but...
        print(archive_path)
        extractor.extract(archive_path=archive_path, cache_root=self.cache_root)

        # Load data
        self._match_graph, self._ref_graph, self._alignment = self._load()
        logging.info(f'Loaded dataset: {self}.')

        logging.info('Starting compaction.')
        self._match_graph, self._ref_graph, self._alignment = _compact(self._match_graph, self._ref_graph, self._alignment)
        logging.info('Finished compaction.')
        logging.info(f'Updated dataset: {self}.')

        for ea in [self._alignment.entity_alignment_train, self._alignment.entity_alignment_test]:
            assert ea[0].max() <= self._match_graph.num_entities
            assert ea[1].max() <= self._ref_graph.num_entities

        if inverse_triples:
            for g in (self._match_graph, self._ref_graph):
                assert not g.inverse_triples
                g.triples, g.relation_label_to_id = add_inverse_triples(g.triples, g.relation_label_to_id)
                g.inverse_triples = True
            logging.info(f'Created inverse triples: {self}')

        if self_loops:
            for g in (self._match_graph, self._ref_graph):
                assert not g.self_loops
                g.triples, g.relation_label_to_id = add_self_loops(g.triples, g.relation_label_to_id)
                g.self_loops = True
            logging.info(f'Created self-loops: {self}')

    @property
    def match_graph(self) -> KnowledgeGraph:
        return self._match_graph

    @property
    def ref_graph(self) -> KnowledgeGraph:
        return self._ref_graph

    @property
    def match_triples(self) -> torch.LongTensor:
        return self._match_graph.triples

    @property
    def ref_triples(self) -> torch.LongTensor:
        return self._ref_graph.triples

    @property
    def entity_alignment_train(self) -> torch.LongTensor:
        return self._alignment.entity_alignment_train

    @property
    def entity_alignment_test(self) -> torch.LongTensor:
        return self._alignment.entity_alignment_test

    @property
    def num_match_triples(self) -> int:
        return self._match_graph.num_triples

    @property
    def num_match_entities(self) -> int:
        return self._match_graph.num_entities

    @property
    def num_match_relations(self) -> int:
        return self._match_graph.num_relations

    @property
    def num_ref_triples(self) -> int:
        return self._ref_graph.num_triples

    @property
    def num_ref_entities(self) -> int:
        return self._ref_graph.num_entities

    @property
    def num_ref_relations(self) -> int:
        return self._ref_graph.num_relations

    @property
    def num_train_alignments(self) -> int:
        return self._alignment.num_train_alignments

    @property
    def num_test_alignments(self) -> int:
        return self._alignment.num_test_alignments

    def _load(self) -> Tuple[KnowledgeGraph, KnowledgeGraph, KGAlignment]:
        match_graph = self._load_graph(match=True)
        ref_graph = self._load_graph(match=False)
        alignment = self._load_alignment(match_graph=match_graph, ref_graph=ref_graph)
        return match_graph, ref_graph, alignment

    def __str__(self):
        return f'{self.__class__.__name__}(match={self._match_graph}, ref={self._ref_graph}, align={self._alignment})'

    def _load_graph(self, match: bool) -> KnowledgeGraph:
        raise NotImplementedError

    def _load_alignment(self, match_graph: KnowledgeGraph, ref_graph: KnowledgeGraph) -> KGAlignment:
        raise NotImplementedError


class DBP15K(KnowledgeGraphAlignmentDataset):
    SPLITS = {str(i) for i in (10, 20, 30, 40, 50)}
    SUBSETS = {'zh_en', 'ja_en', 'fr_en'}
    URL = 'https://github.com/nju-websoft/JAPE/raw/master/data/dbp15k.tar.gz'

    def __init__(
        self,
        subset: str = 'fr_en',
        cache_root: Optional[str] = None,
        inverse_triples: bool = False,
        split: str = '30',
        self_loops: bool = False,
    ):
        if subset not in DBP15K.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed: {DBP15K.SUBSETS}.')
        self.subset = subset

        if split not in DBP15K.SPLITS:
            raise KeyError(f'Unknown split: {split}. Allowed: {DBP15K.SPLITS}.')
        self.split = '0_' + split[0]

        super().__init__(
            url=DBP15K.URL,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'dbp15k', self.subset, self.split)

    def _load_graph(self, match: bool) -> KnowledgeGraph:
        num = 1 if match else 2
        triple_path = path.join(self.root, f'triples_{num}')
        triples: torch.LongTensor = torch.tensor([[int(_id) for _id in row] for row in _load_file(file_path=triple_path)], dtype=torch.long)
        id2e_path = path.join(self.root, f'ent_ids_{num}')
        entity_to_id = {entity: int(_id) for _id, entity in _load_file(id2e_path)}
        id2r_path = path.join(self.root, f'rel_ids_{num}')
        relation_to_id = {rel: int(_id) for _id, rel in _load_file(id2r_path)}

        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_to_id,
            relation_label_to_id=relation_to_id,
        )

    def _load_alignment(self, match_graph: KnowledgeGraph, ref_graph: KnowledgeGraph) -> KGAlignment:
        ea_train, ea_test, ra_train = [
            torch.tensor([
                [int(e) for e in row] for row in _load_file(file_path=path.join(self.root, fp))
            ], dtype=torch.long).t()
            for fp in ['sup_ent_ids', 'ref_ent_ids', 'sup_rel_ids']
        ]
        return KGAlignment(
            entity_alignment_train=ea_train,
            entity_alignment_test=ea_test,
            relation_alignment_train=ra_train,
            relation_alignment_test=None,
        )


class DBP15KFull(KnowledgeGraphAlignmentDataset):
    SUBSETS = {'zh_en', 'ja_en', 'fr_en'}
    URL = 'http://ws.nju.edu.cn/jape/data/DBP15k.tar.gz'

    def __init__(
        self,
        subset: str = 'fr_en',
        cache_root: Optional[str] = None,
        split: float = '30',
        inverse_triples: bool = False,
        self_loops: bool = False,
        random_seed: int = 42,
    ):
        if subset not in DBP15KFull.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed: {DBP15KFull.SUBSETS}.')
        self.subset = subset

        self.split = int(split) * 0.01

        if self.split <= 0. or self.split >= 1.:
            raise KeyError(f'Split must be a float with 0 < split < 1.')

        self.random_seed = random_seed

        super().__init__(
            url=DBP15KFull.URL,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'DBP15k', self.subset)

    def _load_graph(self, match: bool) -> KnowledgeGraph:
        if match:
            match_lang = self.subset.split('_')[0]
            triples_path = path.join(self.root, f'{match_lang}_rel_triples')
        else:
            triples_path = path.join(self.root, 'en_rel_triples')
        triples, entity_label_to_id, relation_label_to_id = load_triples(triples_path=triples_path, delimiter='\t')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
        )

    def _load_alignment(self, match_graph: KnowledgeGraph, ref_graph: KnowledgeGraph) -> KGAlignment:
        entity_alignment_path = path.join(self.root, 'ent_ILLs')
        entity_alignment = _load_alignment(
            alignment_path=entity_alignment_path,
            left_label_to_id=match_graph.entity_label_to_id,
            right_label_to_id=ref_graph.entity_label_to_id,
        )

        # Random split
        generator: torch.Generator = torch.Generator(device=entity_alignment.device).manual_seed(self.random_seed)
        num_alignments = entity_alignment.shape[1]
        perm = torch.randperm(num_alignments, generator=generator)
        last_train_idx = int(self.split * num_alignments)
        entity_alignment_train = entity_alignment.t()[perm[:last_train_idx]].t()
        entity_alignment_test = entity_alignment.t()[perm[last_train_idx:]].t()

        return KGAlignment(
            entity_alignment_train=entity_alignment_train,
            entity_alignment_test=entity_alignment_test,
            relation_alignment_train=None,
            relation_alignment_test=None,
        )


class DWY100K(KnowledgeGraphAlignmentDataset):
    SUBSETS = {'wd', 'yg'}
    URL = 'https://drive.google.com/open?id=1AvLxawvI7J0oFhCtp2il7j7bBUDonBbr'

    def __init__(
        self,
        subset: str = 'wd',
        cache_root: Optional[str] = None,
        inverse_triples: bool = False,
        self_loops: bool = False,
        random_seed: int = 42,
    ):
        if subset not in DWY100K.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed: {DWY100K.SUBSETS}.')
        self.subset = subset

        self.split = 0.3

        self.random_seed = random_seed

        super().__init__(
            url=DWY100K.URL,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
            extractor=ZipExtractor(),
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'DWY100K', f'dbp_{self.subset}', 'mapping', '0_3')

    def _load_graph(self, match: bool) -> KnowledgeGraph:
        num = 1 if match else 2
        triple_path = path.join(self.root, f'triples_{num}')
        triples: torch.LongTensor = torch.tensor([[int(_id) for _id in row] for row in _load_file(file_path=triple_path)], dtype=torch.long)
        id2e_path = path.join(self.root, f'ent_ids_{num}')
        entity_to_id = {entity: int(_id) for _id, entity in _load_file(id2e_path)}
        id2r_path = path.join(self.root, f'rel_ids_{num}')
        relation_to_id = {rel: int(_id) for _id, rel in _load_file(id2r_path)}

        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_to_id,
            relation_label_to_id=relation_to_id,
        )

    def _load_alignment(self, match_graph: KnowledgeGraph, ref_graph: KnowledgeGraph) -> KGAlignment:
        ea_train, ea_test = [
            torch.tensor([
                [int(e) for e in row] for row in _load_file(file_path=path.join(self.root, fp))
            ], dtype=torch.long).t()
            for fp in ['sup_ent_ids', 'ref_ent_ids']
        ]
        return KGAlignment(
            entity_alignment_train=ea_train,
            entity_alignment_test=ea_test,
            relation_alignment_train=None,
            relation_alignment_test=None,
        )


class WK3l(KnowledgeGraphAlignmentDataset):
    URL = 'https://drive.google.com/open?id=1AsPPU4ka1Rc9u-XYMGWtvV65hF3egi0z'
    SUBSETS = {'en_de', 'en_fr'}
    SIZES = {'15k', '120k'}

    def __init__(
        self,
        cache_root: Optional[str] = None,
        subset: str = 'en_de',
        size: str = '15k',
        split: Optional[float] = 0.3,
        inverse_triples: bool = False,
        self_loops: bool = False,
    ):
        if subset not in WK3l.SUBSETS:
            raise KeyError(f'Unknown subset: {subset}. Allowed are: {WK3l.SUBSETS}.')
        self.subset = subset

        if size not in WK3l.SIZES:
            raise KeyError(f'Unknown size: {size}. Allowed are: {WK3l.SIZES}.')
        self.size = size

        if split is None:
            split = .3
        self.split = split

        super().__init__(
            url=WK3l.URL,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
            extractor=ZipExtractor(),
        )

    @property
    def root(self) -> str:
        return path.join(self.cache_root, 'data', f'WK3l-{self.size}', self.subset)

    def _load_graph(self, match: bool) -> KnowledgeGraph:
        lang_match, lang_ref = self.subset.split('_')
        lang = lang_match if match else lang_ref
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        triples_path = path.join(self.root, f'P_{lang}_v{suffix}.csv')
        triples, entity_label_to_id, relation_label_to_id = load_triples(triples_path=triples_path, delimiter='@@@')
        return KnowledgeGraph(
            triples=triples,
            entity_label_to_id=entity_label_to_id,
            relation_label_to_id=relation_label_to_id,
        )

    def _load_alignment(self, match_graph: KnowledgeGraph, ref_graph: KnowledgeGraph) -> KGAlignment:
        # From first ILLs
        lang_a, lang_b = self.subset.split('_')
        suffix = '' if self.size == '15k' else '_' + str(self.size)
        first_ill_path = path.join(self.root, f'{lang_a}2{lang_b}_fk{suffix}.csv')
        first_ill_label_alignment = _load_file(file_path=first_ill_path, delimiter='@@@')
        first_ill_id_alignment = _label_alignment_to_id_alignment(
            first_ill_label_alignment,
            match_graph.entity_label_to_id,
            ref_graph.entity_label_to_id,
        )
        logging.info(f'Loaded alignment of size {len(first_ill_id_alignment)} from first ILL: {first_ill_path}.')

        # From second ILLs
        second_ill_path = path.join(self.root, f'{lang_b}2{lang_a}_fk{suffix}.csv')
        second_ill_label_alignment = _load_file(file_path=second_ill_path, delimiter='@@@')
        second_ill_id_alignment = _label_alignment_to_id_alignment(
            second_ill_label_alignment,
            ref_graph.entity_label_to_id,
            match_graph.entity_label_to_id,
        )
        second_ill_id_alignment = set(tuple(reversed(a)) for a in second_ill_id_alignment)
        logging.info(f'Loaded alignment of size {len(second_ill_id_alignment)} from second ILL: {second_ill_path}.')

        # Load label alignment
        version = 5 if self.subset == 'en_fr' else 6
        suffix = f'{version}' if self.size == '15k' else f'{version}_{self.size}'
        triple_alignment_path = path.join(self.root, f'P_{self.subset}_v{suffix}.csv')
        triple_alignment = _load_file(file_path=triple_alignment_path, delimiter='@@@')

        # From triples
        subject_alignment = set((row[0], row[3]) for row in triple_alignment)
        object_alignment = set((row[2], row[5]) for row in triple_alignment)
        entity_alignment = subject_alignment.union(object_alignment)
        triple_id_alignment = _label_alignment_to_id_alignment(
            entity_alignment,
            match_graph.entity_label_to_id,
            ref_graph.entity_label_to_id,
        )
        logging.info(f'Loaded alignment of size {len(triple_id_alignment)} from triple alignment: {triple_alignment_path}.')

        # Merge alignments
        id_alignment = first_ill_id_alignment.union(second_ill_id_alignment).union(triple_id_alignment)
        logging.info(f'Merged alignments to alignment of size {len(id_alignment)}.')

        # As he split used by MTransE (ILL for testing, triples aligments for training) contains more than 95% test leakage, we use our own split
        sorted_id_alignment = numpy.asarray(sorted(id_alignment))
        assert sorted_id_alignment.shape[1] == 2
        rnd = numpy.random.RandomState(seed=42)
        rnd.shuffle(sorted_id_alignment)
        split_idx = int(numpy.round(self.split * len(sorted_id_alignment)))
        entity_alignment_train: torch.LongTensor = torch.tensor(sorted_id_alignment[:split_idx, :].T, dtype=torch.long)
        entity_alignment_test: torch.LongTensor = torch.tensor(sorted_id_alignment[split_idx:, :].T, dtype=torch.long)
        logging.info(f'Split alignments to {100 * self.split:2.2f}% train equal to size {entity_alignment_train.shape[1]},'
                     f'and {100 * (1. - self.split):2.2f}% test equal to size {entity_alignment_test.shape[1]}.')

        alignment = KGAlignment(
            entity_alignment_train=entity_alignment_train,
            entity_alignment_test=entity_alignment_test,
            relation_alignment_train=None,
            relation_alignment_test=None
        )

        return alignment


def get_dataset_by_name(
    dataset_name: str,
    subset_name: Optional[str] = None,
    cache_root: Optional[str] = None,
    inverse_triples: bool = False,
    self_loops: bool = False,
    split: Union[None, str, float] = None,
    **kwargs
) -> KnowledgeGraphAlignmentDataset:
    """Load a dataset specified by name and subset name.

    :param dataset_name:
        The case-insensitive dataset name. One of ("DBP15k", )
    :param subset_name:
        An optional subset name
    :param cache_root:
        An optional cache directory for extracted downloads. If None is given, use /tmp/{dataset_name}
    :param inverse_triples:
        Whether to generate inverse triples (o, p_inv, s) for every triple (s, p, o).
    :param self_loops:
        Whether to generate self-loops (e, self_loop, e) for each entity e.
    :param split:
        A specification of the train-test split to use.
    :param kwargs:
        Additional key-word based arguments passed to the individual datasets.

    :return: A dataset.
    """
    dataset_name = dataset_name.lower()
    if dataset_name == 'dbp15k_full':
        dataset = DBP15KFull(
            subset=subset_name,
            cache_root=cache_root,
            split=split,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
        )
    elif dataset_name == 'dbp15k_jape':
        dataset = DBP15K(
            subset=subset_name,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            split=split,
            self_loops=self_loops,
        )
    elif 'wk3l' in dataset_name:
        # size = kwargs.pop('size', '15k')
        size = re.search('wk3l([0-9]+k)', dataset_name).group(1)
        if split == '30':
            split = 0.3
        dataset = WK3l(
            cache_root=cache_root,
            subset=subset_name,
            size=size,
            inverse_triples=inverse_triples,
            split=split,
            self_loops=self_loops,
        )
    elif dataset_name == 'dwy100k':
        dataset = DWY100K(
            subset=subset_name,
            cache_root=cache_root,
            inverse_triples=inverse_triples,
            self_loops=self_loops,
        )
    else:
        raise KeyError(f'Could not load dataset: "{dataset_name}"')
    return dataset


def _load_alignment(
    alignment_path: Union[str, TextIO],
    left_label_to_id: Mapping[str, int],
    right_label_to_id: Mapping[str, int],
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> torch.LongTensor:
    # Load label alignment
    label_alignment = _load_file(file_path=alignment_path, delimiter=delimiter, encoding=encoding)

    alignment = _label_alignment_to_id_alignment(
        label_alignment,
        left_label_to_id,
        right_label_to_id,
    )
    alignment = torch.tensor(list(zip(*alignment)), dtype=torch.long)

    logging.info(f'Loaded alignment of size {alignment.shape[1]}')

    return alignment


def _label_alignment_to_id_alignment(
    label_array: Collection[Collection[str]],
    *column_label_to_ids: Mapping[str, int],
) -> Set[Collection[int]]:
    num_raw = len(label_array)

    # Drop duplicates
    label_array = set(map(tuple, label_array))
    num_without_duplicates = len(label_array)
    if num_without_duplicates < num_raw:
        logging.warning(f'Dropped {num_raw - num_without_duplicates} duplicate rows.')

    # Translate to id
    result = {tuple(l2i.get(e, None) for l2i, e in zip(column_label_to_ids, row)) for row in label_array}
    before_filter = len(result)
    result = set(row for row in result if None not in row)
    after_filter = len(result)
    logging.info(f'Translated list of length {before_filter} to label array of length {after_filter}.')

    return result


def load_triples(
    triples_path: Union[str, TextIO],
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> Tuple[torch.LongTensor, Mapping[str, int], Mapping[str, int]]:
    """Load triples."""
    # Load triples from tsv file
    label_triples = _load_file(file_path=triples_path, delimiter=delimiter, encoding=encoding)

    # Split
    heads, relations, tails = [[t[i] for t in label_triples] for i in range(3)]

    # Sorting ensures consistent results when the triples are permuted
    entity_label_to_id = {
        e: i for i, e in enumerate(sorted(set(heads).union(tails)))
    }
    relation_label_to_id = {
        r: i for i, r in enumerate(sorted(set(relations)))
    }

    id_triples = _label_alignment_to_id_alignment(
        label_triples,
        entity_label_to_id,
        relation_label_to_id,
        entity_label_to_id,
    )
    triples: torch.LongTensor = torch.tensor(list(id_triples), dtype=torch.long)

    # Log some info
    num_triples = triples.shape[0]
    num_entities = len(entity_label_to_id)
    num_relations = len(relation_label_to_id)
    logging.info(f'Loaded {num_triples} unique triples, '
                 f'with {num_entities} unique entities, '
                 f'and {num_relations} unique relations.')

    return triples, entity_label_to_id, relation_label_to_id


def _load_file(
    file_path: str,
    delimiter: str = '\t',
    encoding: str = 'utf8',
) -> List[List[str]]:
    with open(file_path, 'r', encoding=encoding) as f:
        out = [line[:-1].split(sep=delimiter) for line in f.readlines()]
    return out
