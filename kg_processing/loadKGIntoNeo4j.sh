#!/bin/bash 
 
export JAVA_HOME=PATH/TO/JDK
export PATH=$JAVA_HOME/bin:$PATH
export NEO4J_HOME=PATH/TO/NEO4J 
export CONQUER_HOME=PATH/TO/CONQUER
 
$NEO4J_HOME/bin/neo4j-admin import --id-type=STRING --nodes=e="node_headers.csv,$CONQUER_HOME/data/dumps/nodes_entities.csv" --nodes="node_headers.csv,$CONQUER_HOME/data/dumps/nodes_literals.csv" --relationships=triples="edge_headers.csv,$CONQUER_HOME/data/dumps/triple_only_dump.csv" --relationships=s_o_qualifiers="edge_headers.csv,$CONQUER_HOME/data/dumps/s_o_qualifiers_dump.csv" --relationships=s_qualifierobj="edge_headers.csv,$CONQUER_HOME/data/dumps/s_qualifierobj_dump.csv" --relationships=o_qualifierobj="edge_headers.csv,$CONQUER_HOME/data/dumps/o_qualifierobj_dump.csv" --relationships=qo_qo="edge_headers.csv,$CONQUER_HOME/data/dumps/qo_qo_dump.csv" --verbose --high-io=true --legacy-style-quoting=true --skip-bad-relationships=true --skip-duplicate-nodes=true --bad-tolerance=1000000000000 --delimiter=","

