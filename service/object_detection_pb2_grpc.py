# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import message.object_detection_pb2 as object__detection__pb2


class DetectionStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.DetectObjects = channel.unary_unary(
                '/godeye_service.object_detection.Detection/DetectObjects',
                request_serializer=object__detection__pb2.DetectRequest.SerializeToString,
                response_deserializer=object__detection__pb2.DetectResponse.FromString,
                )


class DetectionServicer(object):
    """Missing associated documentation comment in .proto file."""

    def DetectObjects(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DetectionServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'DetectObjects': grpc.unary_unary_rpc_method_handler(
                    servicer.DetectObjects,
                    request_deserializer=object__detection__pb2.DetectRequest.FromString,
                    response_serializer=object__detection__pb2.DetectResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'godeye_service.object_detection.Detection', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Detection(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def DetectObjects(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/godeye_service.object_detection.Detection/DetectObjects',
            object__detection__pb2.DetectRequest.SerializeToString,
            object__detection__pb2.DetectResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)