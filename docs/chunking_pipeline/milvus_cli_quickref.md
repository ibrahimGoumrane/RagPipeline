# Milvus CLI — Quick Reference
Connecting to your local Milvus instance and inspecting collections, schemas, and data.

---

## 1. Launch the CLI

```shell
milvus_cli
```

You should see the `milvus_cli >` prompt appear.

---

## 2. Connect to your Milvus instance

Since you installed via Docker, Milvus is running on localhost port 19530 by default.

```shell
milvus_cli > connect -uri http://127.0.0.1:19530
```

If you set a username and password during setup:

```shell
milvus_cli > connect -uri http://host.docker.internal:19530 -t root:Milvus
```

---

## 3. Check your databases

List all databases:

```shell
milvus_cli > list databases
```

Switch to your project database:

```shell
milvus_cli > use database -db your_db_name
```

Show details of the current database:

```shell
milvus_cli > show database
```

---

## 4. Inspect your collections

List all collections in the current database:

```shell
milvus_cli > list collections
```

Show the full schema and details of a specific collection (replace `rag_chunks_children` with your actual collection name):

```shell
milvus_cli > show collection -c rag_chunks_children
```

Show collection statistics (number of entities, etc.):

```shell
milvus_cli > show collection_stats -c rag_chunks_children
```

Do the same for your parent collection:

```shell
milvus_cli > show collection -c rag_chunks_parent
milvus_cli > show collection_stats -c rag_chunks_parent
```

---

## 5. Inspect your indexes

List all indexes on a collection:

```shell
milvus_cli > list indexes -c rag_chunks_children
```

Show the details of a specific index (replace `embedding` with your index name if different):

```shell
milvus_cli > show index -c rag_chunks_children -in embedding
```

Check indexing progress:

```shell
milvus_cli > show index_progress -c rag_chunks_children
```

Check load state (whether the collection is loaded into RAM):

```shell
milvus_cli > show load_state -c rag_chunks_children
```

---

## 6. Query and inspect your data

Get specific entities by ID:

```shell
milvus_cli > get

Collection name: rag_chunks_children
IDs (comma-separated): your_chunk_id_here
Output fields (comma-separated, or * for all) []: *
```

Query entities with a filter — for example all chunks belonging to a specific document:

```shell
milvus_cli > query

Collection name: rag_chunks_children
The query expression: doc_id == "document_001"
Name of partitions that contain entities []: 
A list of fields to return []: chunk_id, doc_id, parent_id, content, page_ref
timeout []: 
Guarantee timestamp [0]: 
Graceful time [5]: 
```

Query the parent collection:

```shell
milvus_cli > query

Collection name: rag_chunks_parent
The query expression: doc_id == "document_001"
Name of partitions that contain entities []: 
A list of fields to return []: parent_id, doc_id, heading, page_no
timeout []: 
Guarantee timestamp [0]: 
Graceful time [5]: 
```

---

## 7. Change output format

By default results print as a table. Switch to JSON if you want the full raw data:

```shell
milvus_cli > set output json
```

Switch back to table:

```shell
milvus_cli > set output table
```

---

## 8. Exit

```shell
milvus_cli > exit
```

---

## Quick cheat sheet

| What you want to do | Command |
|---|---|
| Connect | `connect -uri http://127.0.0.1:19530` |
| List databases | `list databases` |
| Switch database | `use database -db <name>` |
| List collections | `list collections` |
| See schema | `show collection -c <name>` |
| Count entities | `show collection_stats -c <name>` |
| See indexes | `list indexes -c <name>` |
| Check index detail | `show index -c <name> -in <index_name>` |
| Check load state | `show load_state -c <name>` |
| Query by filter | `query` (interactive) |
| Get by ID | `get` (interactive) |
| JSON output | `set output json` |
| Exit | `exit` |
