/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Modifications Copyright (c) 2025-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"

ncclResult_t ncclTransportRingConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
  }

  VERBOSE(NCCL_ALLOC, "comm %p Ring totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  VERBOSE(NCCL_ALLOC, "comm %p Ring totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_INIT, "Connected all rings");
exit:

  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    VERBOSE(NCCL_ALLOC, "comm %p Tree totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p Tree totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportPatConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int mask = 1; mask < comm->nRanks; mask <<= 1) {
      int prevPeer = (comm->rank + mask) % comm->nRanks;
      int nextPeer = (comm->rank + comm->nRanks - mask) % comm->nRanks;
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        struct ncclChannel* channel = comm->channels + c;
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
  }
  VERBOSE(NCCL_ALLOC, "comm %p PAT totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  VERBOSE(NCCL_ALLOC, "comm %p PAT totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  INFO(NCCL_INIT, "Connected binomial trees");
exit:
  return ret;
fail:
  goto exit;
}

// All ranks must participate in collNetSetup call
// The same channel owns both directions: recv connection 0 and send connection 1.
// All ranks participate; the local chain head shares its receive handle with send.
static ncclResult_t collNetSetup(ncclComm_t comm, struct ncclTopoGraph* collNetGraph,
    struct ncclChannel* channel, int masterRank, int type, ncclConnect* connect) {
  ncclResult_t ret = ncclSuccess;
  const int rank = comm->rank, nranks = comm->nRanks, nMasters = comm->nNodes;
  const bool isMaster = rank == masterRank;
  struct ncclPeerInfo* myInfo = comm->peerInfo + rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo + nranks;
  peerInfo->rank = nranks;
  struct ncclChannelPeer* root = channel->peers[nranks];
  struct ncclConnector* conn = type == collNetRecv ? root->recv + type : root->send + type;
  struct ncclTransportComm* transportComm = type == collNetRecv ? &collNetTransport.recv : &collNetTransport.send;
  conn->transportComm = transportComm;
  struct ncclConnect myConnect = {};
  struct {
    int isMaster;
    ncclConnect connect;
  } *allConnects = nullptr;
  ncclConnect* masterConnects = nullptr;
  if (isMaster) {
    NCCLCHECK(transportComm->setup(comm, collNetGraph, myInfo, peerInfo, &myConnect, conn, channel->id, type));
  }
  NCCLCHECKGOTO(ncclCalloc(&masterConnects, nMasters), ret, cleanup);
  if (type == collNetRecv) {
    NCCLCHECKGOTO(ncclCalloc(&allConnects, nranks), ret, cleanup);
    allConnects[rank].isMaster = isMaster;
    allConnects[rank].connect = myConnect;
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allConnects, sizeof(*allConnects)), ret, cleanup);
    int c = 0;
    for (int r = 0; r < nranks; ++r) {
      if (allConnects[r].isMaster) masterConnects[c++] = allConnects[r].connect;
    }
  } else if (isMaster) {
    masterConnects[comm->node] = *connect;
  }
  if (isMaster) {
    NCCLCHECKGOTO(transportComm->connect(comm, masterConnects, nMasters, comm->node, conn), ret, cleanup);
    struct ncclDevChannelPeer* devRoot = channel->devPeersHostPtr[nranks];
    struct ncclConnInfo* devConn = type == collNetRecv ? devRoot->recv + type : devRoot->send + type;
    HGGCCHECKGOTO(hggcMemcpy(devConn, &conn->conn, sizeof(struct ncclConnInfo), hggcMemcpyHostToDevice), ret, cleanup);
    if (type == collNetRecv) *connect = masterConnects[comm->node];
  }
cleanup:
  NCCLCHECK(ncclFree(allConnects));
  NCCLCHECK(ncclFree(masterConnects));
  return ret;
}

static ncclResult_t checkCollNetSetup(ncclComm_t comm, int rank, int collNetSetupFail) {
  ncclResult_t res = ncclSuccess;
  int nranks = comm->nRanks;
  // AllGather collNet setup results
  int* allGatherFailures = NULL;
  NCCLCHECKGOTO(ncclCalloc(&allGatherFailures, nranks), res, exit);
  allGatherFailures[rank] = collNetSetupFail;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGatherFailures, sizeof(int)), res, exit);
  for (int i=0; i<nranks; i++) {
    if (allGatherFailures[i] != 0) {
      collNetSetupFail = 1;
      break;
    }
  }
  if (collNetSetupFail) {
    if (rank == 0) WARN("Cannot initialize CollNet, using %s instead", ncclNetName());
    // Free collNet resources
    for (int r=0; r<comm->nChannels; r++) {
      struct ncclChannel* channel = comm->channels+r;
      struct ncclChannelPeer* peer = channel->peers[nranks];
      for (int b=0; b<NCCL_MAX_CONNS; b++) {
        struct ncclConnector* send = peer->send + b;
        if (send->transportResources && send->transportComm) NCCLCHECK(send->transportComm->free(send));
        send->transportResources = NULL; // avoid double free
      }
      for (int b=0; b<NCCL_MAX_CONNS; b++) {
        struct ncclConnector* recv = peer->recv + b;
        if (recv->transportResources && recv->transportComm) NCCLCHECK(recv->transportComm->free(recv));
        recv->transportResources = NULL; // avoid double free
      }
    }
    // Set support to 0
    comm->collNetSupport = 0;
  } else {
    comm->collNetSupport = 1;
  }
exit:
  NCCLCHECK(ncclFree(allGatherFailures));
  return res;
}

ncclResult_t ncclTransportCollnetConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    struct ncclTopoGraph *collNetGraph = &comm->graphs[NCCL_ALGO_COLLNET];
    int collNetSetupFail = 0;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      // The receive and send network endpoint belong to the same chain head.
      const int master = collNetGraph->intra[(c % collNetGraph->nChannels)*comm->localRanks];
      ncclConnect connect = {};
      if (collNetSetup(comm, collNetGraph, channel, master, collNetRecv, &connect) != ncclSuccess)
        collNetSetupFail = 1;
      if (collNetSetup(comm, collNetGraph, channel, master, collNetSend, &connect) != ncclSuccess)
        collNetSetupFail = 1;
    }
    NCCLCHECKGOTO(checkCollNetSetup(comm, comm->rank, collNetSetupFail), ret, fail);
    if (!comm->collNetSupport) goto exit;
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, &channel->collnetChain.up,
          1, channel->collnetChain.down, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 0), ret, fail);
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, 1, channel->collnetChain.down,
          1, &channel->collnetChain.up, 1), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 1), ret, fail);

    VERBOSE(NCCL_ALLOC, "comm %p CollNet totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d",
      comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p CollNet totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d",
      comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
  }
exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportFcConnect(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t devMemUsedStart = comm->devMemUsed;
  uint64_t pinnedMemUsedStart = comm->pinnedMemUsed;
  uint64_t sharedDevMemUsedStart = comm->sharedRes->devMemUsed;
  uint64_t sharedPinnedMemUsedStart = comm->sharedRes->pinnedMemUsed;

  if (comm && comm->nRanks > 1) {
    for (int c = 0; c < comm->fcnChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, channel, NCCL_MAX_FC_ARITY, channel->fc.peerRanks,
            NCCL_MAX_FC_ARITY, channel->fc.peerRanks, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_FC], 0), ret, fail);
    VERBOSE(NCCL_ALLOC, "comm %p Fc totally used %ldB %ldB(sharedRes) device memory splitShare %d nranks %d rank %d", comm, comm->devMemUsed - devMemUsedStart, comm->sharedRes->devMemUsed - sharedDevMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    VERBOSE(NCCL_ALLOC, "comm %p Fc totally used %ldB %ldB(sharedRes) pinned memory splitShare %d nranks %d rank %d", comm, comm->pinnedMemUsed - pinnedMemUsedStart, comm->sharedRes->pinnedMemUsed - sharedPinnedMemUsedStart, comm->config.splitShare, comm->nRanks, comm->rank);
    INFO(NCCL_INIT, "Connected all fc graphs");
  }
exit:
  return ret;
fail:
  goto exit;
}
