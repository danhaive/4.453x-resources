import json
import numpy as np
import os


def node_to_dictionary(node: np.array,node_id: int, is_fixed: bool) -> dict: 
    return {
        "point":{
            "X":node[0],
            "Y":node[1],
            "Z":node[2],

        },
        "node_id":node_id,
        "is_grounded": int(is_fixed),
        "fixities": 3*[int(is_fixed)]
    }

def element_to_dictionary(element: np.ndarray, element_id: int,):
    return {
        "end_node_ids":[int(element[0]), int(element[1])],
        "element_id": element_id,
        "layer_id": 0
    }



def save_truss_as_json(nodes: np.ndarray, edges: np.ndarray, support_indices: np.ndarray, file_name: str, dir_name: str = "generated_truss_jsons"):
    
    node_list = []
    element_list = []
    for i, node in enumerate(nodes):
        node_list.append(node_to_dictionary(node,i,i in support_indices))
    for i, element in enumerate(edges):
        element_list.append(element_to_dictionary(element, i))

    truss = {"node_list": node_list, "element_list": element_list }

    os.makedirs(dir_name, exist_ok=True)
    with open(f'{dir_name}/{file_name}', 'w') as f:
        print("saving")
        json.dump(truss, f)


def unnormalize(values, bounds):
    return values * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


if __name__=="__main__":

    import pyDOE as doe
    from utils.geometry import generate_space_truss_2
    from tqdm import tqdm
    from scipy.spatial import KDTree
    np.random.seed(2)

    n_samples = 100
    design_space_dim = 6
    bounds = [[-0.5, 3] * design_space_dim]
    normalized_lhs_samples = doe.lhs(design_space_dim, n_samples)    
    lhs_samples = unnormalize(normalized_lhs_samples, np.array(bounds))


    # geometry constants
    column_locations = [[-3, -3, -3], [-1, -3, -3], [-1, 3, -3], [-3, 3, -3]]
    width = 10
    n_modules = 10

    for i, sample in tqdm(enumerate(lhs_samples)):
        nodes, edges = generate_space_truss_2(
            sample[0],
            sample[1],
            sample[2],
            sample[3],
            sample[4],
            sample[5],
            width=10,
            n_modules=10,
            column_locations=column_locations,
        )

            # get support indices
        kd_tree = KDTree(nodes)
        _, support_indices = kd_tree.query(column_locations)

        save_truss_as_json(nodes, edges, support_indices, f"space_truss_{str(i).zfill(5)}.json")


    