# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import message.person_reid_pb2 as person__reid__pb2


class PersonReIDStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetFeature = channel.unary_unary(
                '/godeye_service.person_reid.PersonReID/GetFeature',
                request_serializer=person__reid__pb2.FeatureRequest.SerializeToString,
                response_deserializer=person__reid__pb2.FeatureResponse.FromString,
                )
        self.GetFeatures = channel.stream_stream(
                '/godeye_service.person_reid.PersonReID/GetFeatures',
                request_serializer=person__reid__pb2.FeatureRequest.SerializeToString,
                response_deserializer=person__reid__pb2.FeatureResponse.FromString,
                )


class PersonReIDServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetFeature(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFeatures(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_PersonReIDServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetFeature': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFeature,
                    request_deserializer=person__reid__pb2.FeatureRequest.FromString,
                    response_serializer=person__reid__pb2.FeatureResponse.SerializeToString,
            ),
            'GetFeatures': grpc.stream_stream_rpc_method_handler(
                    servicer.GetFeatures,
                    request_deserializer=person__reid__pb2.FeatureRequest.FromString,
                    response_serializer=person__reid__pb2.FeatureResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'godeye_service.person_reid.PersonReID', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class PersonReID(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetFeature(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/godeye_service.person_reid.PersonReID/GetFeature',
            person__reid__pb2.FeatureRequest.SerializeToString,
            person__reid__pb2.FeatureResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFeatures(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/godeye_service.person_reid.PersonReID/GetFeatures',
            person__reid__pb2.FeatureRequest.SerializeToString,
            person__reid__pb2.FeatureResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)