# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file to You under the
# License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.

import asyncio
from typing import Any, Dict

from tqdm import tqdm

from hugegraph_llm.config import huge_settings
from hugegraph_llm.indices.vector_index.base import VectorStoreBase
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from hugegraph_llm.operators.hugegraph_op.schema_manager import SchemaManager
from hugegraph_llm.utils.log import log


class BuildSemanticIndex:
    def __init__(self, embedding: BaseEmbedding, vector_index: type[VectorStoreBase]):
        self.vid_index = vector_index.from_name(embedding.get_embedding_dim(), huge_settings.graph_name, "graph_vids")
        self.embedding = embedding
        self.sm = SchemaManager(huge_settings.graph_name)

    def _extract_names(self, vertices: list) -> list[str]:
        """Extract primary key values from composite vertex IDs (format: 'label:pk_value')."""
        result = []
        for v in vertices:
            if not isinstance(v, str) or ":" not in v:
                log.warning("Skipping non-primary-key vertex ID: %s (type: %s)", v, type(v).__name__)
                continue
            result.append(v.split(":")[1])
        return result

    def _fetch_pk_values(self, vids: list) -> Dict[str, str]:
        """Fetch primary key values from the graph for given vertex IDs.

        This is needed for backends (e.g. HBase) where Gremlin g.V().id()
        returns internal numeric IDs instead of composite "label:pk_value" strings,
        and g.V(specific_id) lookups may not work.
        """
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        label_pk_map = {vl["name"]: vl.get("primary_keys", [None])[0] for vl in vertexlabels}

        if not label_pk_map:
            return {}

        vids_set = set(str(v) for v in vids)
        result = {}
        try:
            # Use g.V() without ID filter — on some backends (e.g. HBase),
            # g.V(specific_id) lookups fail even though g.V().id() returns those IDs
            data = self.sm.client.gremlin().exec("g.V().valueMap(true).limit(10000).toList()")["data"]
            for item in data:
                vid = str(item.get("id"))
                if vid not in vids_set:
                    continue
                label = item.get("label")
                pk_name = label_pk_map.get(label)
                if pk_name and pk_name in item:
                    pk_values = item[pk_name]
                    if pk_values:
                        result[vid] = str(pk_values[0])
        except Exception as e:  # pylint: disable=broad-exception-caught
            log.error("Failed to fetch primary key values from graph: %s", e)
        return result

    async def _get_embeddings_parallel(self, vids: list[str]) -> list[Any]:
        sem = asyncio.Semaphore(10)
        batch_size = 1000

        async def get_embeddings_with_semaphore(vid_list: list[str]) -> Any:
            async with sem:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.embedding.get_texts_embeddings, vid_list)

        vid_batches = [vids[i : i + batch_size] for i in range(0, len(vids), batch_size)]
        tasks = [get_embeddings_with_semaphore(batch) for batch in vid_batches]

        embeddings = []
        with tqdm(total=len(tasks)) as pbar:
            for future in asyncio.as_completed(tasks):
                batch_embeddings = await future
                embeddings.extend(batch_embeddings)
                pbar.update(1)
        return embeddings

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        vertexlabels = self.sm.schema.getSchema()["vertexlabels"]
        all_pk_flag = bool(vertexlabels) and all(data.get("id_strategy") == "PRIMARY_KEY" for data in vertexlabels)

        past_vids = self.vid_index.get_all_properties()
        # TODO: We should build vid vector index separately, especially when the vertices may be very large
        present_vids = context["vertices"]  # Warning: data truncated by fetch_graph_data.py
        # Normalize all IDs to strings for consistent comparison
        present_vids_str = [str(v) for v in present_vids]
        past_vids_str = [str(v) for v in past_vids]

        removed_vids = set(past_vids_str) - set(present_vids_str)
        removed_num = self.vid_index.remove(removed_vids)
        added_vids = list(set(present_vids_str) - set(past_vids_str))

        if added_vids:
            if all_pk_flag:
                # Try extracting names from composite IDs first
                vids_to_process = self._extract_names(added_vids)
                if vids_to_process:
                    added_vids = [v for v in added_vids if isinstance(v, str) and ":" in v]
                else:
                    # Fallback: fetch primary key values directly from graph
                    # (needed for HBase backend where g.V().id() returns integers)
                    log.info("Vertex IDs are not in composite format, fetching pk values from graph...")
                    pk_map = self._fetch_pk_values(added_vids)
                    if pk_map:
                        matched_vids = [v for v in added_vids if v in pk_map]
                        vids_to_process = [pk_map[v] for v in matched_vids]
                        added_vids = matched_vids
                    else:
                        log.warning("Failed to fetch primary key values, skipping embedding.")
                        vids_to_process = []
            else:
                vids_to_process = added_vids

            if vids_to_process:
                added_embeddings = asyncio.run(self._get_embeddings_parallel(vids_to_process))
                log.info("Building vector index for %s vertices...", len(added_vids))
                self.vid_index.add(added_embeddings, added_vids)
                self.vid_index.save_index_by_name(huge_settings.graph_name, "graph_vids")
        else:
            log.debug("No update vertices to build vector index.")
        context.update(
            {
                "removed_vid_vector_num": removed_num,
                "added_vid_vector_num": len(added_vids) if added_vids and vids_to_process else 0,
            }
        )
        return context
