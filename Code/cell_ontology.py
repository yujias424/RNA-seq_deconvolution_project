from onto_lib import load_ontology
from onto_lib import ontology_graph

GO_ONT_CONFIG_ID = '18'
UNIT_OG_ID = '7'

ONTO_PATCH = {
    "add_edges": [
        {
            "source_term": "CL:2000001",    # peripheral blood mononuclear cell
            "target_term": "CL:0000081",    # blood cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000670",    # primordial germ cell
            "target_term": "CL:0002321",    # embryonic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0008001",    # hematopoietic precursor cell
            "target_term": "CL:0011115",    # precursor cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002246",    # peripheral blood stem cell
            "target_term": "CL:0000081",    # blood cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000542",    # lymphocyte
            "target_term": "CL:0000842",    # mononuclear cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000066",    # epithelial cell
            "target_term": "CL:0002371",    # somatic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0001035",    # bone cell
            "target_term": "CL:0002371",    # somatic cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000018",    # spermatid
            "target_term": "CL:0011115",    # precursor cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000017",    # spermatocyte
            "target_term": "CL:0011115",    # precursor cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # macrophage
            "target_term": "CL:0000842",    # mononuclear cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000235",    # macrophage
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000451",    # dendritic cell
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0000236",    # B cell
            "target_term": "CL:0000145",    # professional antigen presenting cell
            "edge_type": "is_a"
        },
        {
            "source_term": "CL:0002371",    # somatic cell
            "target_term": "CL:0000255",    # eukaryotic cell
            "edge_type": "is_a"
        }
    ]
}


def patch_the_ontology(og):
    for edge_info in ONTO_PATCH['add_edges']:
        source_id = edge_info['source_term']
        target_id = edge_info['target_term']
        source_term = None
        target_term = None
        if source_id in og.id_to_term:
            source_term = og.id_to_term[source_id]
        if target_id in og.id_to_term:
            target_term = og.id_to_term[target_id]
        if source_term is None or target_term is None:
            continue
        edge_type = edge_info['edge_type']
        inv_edge_type = "inv_%s" % edge_type
        if edge_type in source_term.relationships:
            source_term.relationships[edge_type].append(target_id)
        else:
            source_term.relationships[edge_type] = [target_id]
        if inv_edge_type in target_term.relationships:
            target_term.relationships[inv_edge_type].append(source_id)
        else:
            target_term.relationships[inv_edge_type] = [source_id]
    return og

ONT_NAME_TO_ONT_ID = {"EFO_CL_DOID_UBERON_CVCL":"17"}
ONT_ID_TO_OG = {x: patch_the_ontology(load_ontology.load(x)[0]) for x in ONT_NAME_TO_ONT_ID.values()}




#################################################################################
# Utility functions below:
##################################################################################

def get_term_name(term_id):
    """
    Get the name of a term for a given term id.
    For example, for the term id 'CL:0000236',
    its term name is 'B cell'.
    """
    og = ONT_ID_TO_OG['17']
    try:
        return og.id_to_term[term_id].name
    except KeyError:
        raise KeyError("The term_id '%s' is not in the ontology" % term_id)

def get_terms_without_children(term_ids):
    """
    Given a list of term id's, return all ids that have no
    children in this set. This can be used to compute leaf
    nodes.
    """
    return ontology_graph.most_specific_terms(
        term_ids,
        ONT_ID_TO_OG['17'],
        sup_relations=["is_a"]
    )

def get_children(term_id):
    """
    For a given term id, return the term id's
    for its children.
    """
    return ONT_ID_TO_OG['17'].id_to_term[term_id].relationships['inv_is_a']

def get_parents(term_id):
    """
    For a given term id, return the term id's
    for its parents.
    """
    return ONT_ID_TO_OG['17'].id_to_term[term_id].relationships['is_a']

def get_descendents(term_id):
    """
    For a given term id, return the term id's
    for its descendents.
    """
    return ONT_ID_TO_OG['17'].recursive_subterms(term_id)

def get_ancestors(term_id):
    """
    For a given term id, return the term id's
    for its ancestors.
    """
    return ONT_ID_TO_OG['17'].recursive_superterms(term_id)

def main():
    #og = the_ontology()
    #og = patch_the_ontology(og)
    #print og.id_to_term['CL:0000081']
    #print og.id_to_term['CL:2000001']
    #print og.id_to_term['CL:0000542']

    children = get_ancestors('CL:0000236')
    for child in children:
        print(get_term_name(child))

if __name__ == "__main__":
    main()
