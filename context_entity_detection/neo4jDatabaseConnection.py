from neo4j import GraphDatabase
import re
import random

random.seed(7)

"""Access to our neo4j KG database"""

class KGENVIRONMENT:
    def __init__(self):
        self.uri =  "bolt://127.0.0.1:7687"
        self.driver = GraphDatabase.driver(self.uri, auth=("neo4j", "admin"), encrypted=False,  max_connection_lifetime=3600*24*30, keep_alive=True)
        with self.driver.session() as session:
            result = session.read_transaction(self.__startKG)
        
        print("KG init done", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.driver.close() 

    def __startKG(self, tx):
        return tx.run("CALL apoc.warmup.run(True,True,True);")

    #get one hop neighborhood for entity with "nodeid"
    def get_one_hop_nodes(self, nodeid):
        with self.driver.session() as session:
            try:
                result = session.read_transaction(self.__fetch_one_hop_nodes, nodeid)
            except Exception as e:
                print(e) 
                print("Warning: An exception occured in the neo4j database; an empty list will be returned!")
                return []
            return result

    #calculate number of neighbors
    # we use number of facts the entity with "nodeid" is present in as subject -> count outgoing edges only
    def get_number_of_neighbors(self, nodeid):
        with self.driver.session() as session:
            result = session.read_transaction(self.__fetch_number_of_neighbors, nodeid)
            return result

    #create cypher query to get outgoing edges from "nodeid"
    def __fetch_number_of_neighbors(self, tx, nodeid):
        try:
            return tx.run("MATCH (n:e {nodeid:'" + nodeid + "'}) RETURN size((n)-->())").single().value()
        except:
            return 0 

    #create cypher query to collect up to 1000 paths starting at entity with "nodeid"
    def __fetch_one_hop_nodes(self, tx, nodeid):
        paths = []
        for record in tx.run("MATCH (n:e {nodeid: '" + nodeid + "'}) CALL apoc.path.expandConfig(n, {   minLevel: 1, maxLevel: 1}) YIELD path RETURN nodes(path) as nodes, relationships(path) as relations LIMIT 1000;"):
            relation = re.split("!|\||_",record['relations'][0]['rel'].strip("!"))
            paths.append([record['nodes'][0]['nodeid'], relation ,record['nodes'][1]['nodeid']])
           
        return paths
    
 