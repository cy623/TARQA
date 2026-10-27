from tarqa.data.metadata import SchemaCatalog
from tarqa.prediction.release_aggregator import ReleaseAggregator


def build_relational_aggregator(paths, prediction_cfg):
    aggregation = prediction_cfg.get('aggregation', {})
    representation = prediction_cfg.get('representation', {})
    backend = aggregation.get('backend', 'sql_statistics')
    graph_representation = representation.get('backend') == 'hetero_gnn'
    if (backend == 'hetero_graph') != graph_representation:
        raise ValueError('hetero_graph aggregation and hetero_gnn representation must be enabled together')
    if backend == 'hetero_graph':
        from tarqa.prediction.row_graph import RowGraphAggregator
        if int(representation.get('num_layers', 2)) != int(aggregation.get('num_layers', 2)):
            raise ValueError('Graph extraction depth differs from propagation depth')
        return RowGraphAggregator(paths, aggregation)
    if backend != 'sql_statistics':
        raise ValueError(f'Unsupported aggregation backend: {backend}')
    return ReleaseAggregator(paths, aggregation)


def build_prediction_runtime(paths,prediction_cfg,question_encoder,*,symbolic_time_adapter_path=None):
    aggregator = build_relational_aggregator(paths,prediction_cfg)
    if prediction_cfg.get('aggregation', {}).get('backend') == 'hetero_graph':
        from tarqa.prediction.node_encoder import build_node_encoder
        aggregator.node_encoder = build_node_encoder(prediction_cfg, question_encoder)
    return SchemaCatalog.from_files(paths.domain,paths.metadata,database_path=paths.database),aggregator
