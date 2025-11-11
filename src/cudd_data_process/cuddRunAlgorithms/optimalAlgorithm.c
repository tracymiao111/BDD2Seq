#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "cudd.h"
#include "bnet.h"
#include "util.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

void print_bdd_variable_order(DdManager *dd) {
    int num_vars = Cudd_ReadSize(dd);  // Total number of variables
    printf("\nCurrent variable ordering in BDD:\n");
        // for (int i = 0; i < num_vars; i++) {
        // if (num_vars[i] != NULL) {
        //     printf("Number of DAG nodes for output %d: %d\n", i, Cudd_DagSize(n[i]));
        // } else {
        //     printf("Output %d BDD node is NULL\n", i);
        // }
    //}
    // printf("flag: show num_vars = %d", num_vars);
    for (int i = 0; i < num_vars; i++) {
        // printf("get into the loop flag");
        int var_index = Cudd_ReadInvPerm(dd, i);  // Get the variable index at position i
        printf("Position %d: Variable index %d\n", i, var_index);
    }
    printf("\n");
}

void calculate_dag_sizes(DdManager *dd, BnetNetwork *network) {
    int noutputs = network->noutputs;
    unsigned int total_dag_size = 0;

    for (int i = 0; i < noutputs; i++) {
        BnetNode *output_node = NULL;
        st_lookup(network->hash, network->outputs[i], (void **) &output_node);

        if (output_node != NULL && output_node->dd != NULL) {
            unsigned int dag_size = Cudd_DagSize(output_node->dd);
            printf("DAG size for output %d (%s): %u\n", i, network->outputs[i], dag_size);
            total_dag_size += dag_size;
        } else {
            printf("Error: Output node BDD for index %d not found or not built.\n", i);
        }
    }

    printf("Total DAG size (sum of all output DAG sizes): %u\n", total_dag_size);
}

void calculate_total_sharing_size(DdManager *dd, BnetNetwork *network) {
    int noutputs = network->noutputs;
    DdNode **bdd_outputs = (DdNode **)malloc(noutputs * sizeof(DdNode *));
    
    if (bdd_outputs == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory for output BDD array.\n");
        return;
    }

    for (int i = 0; i < noutputs; i++) {
        BnetNode *output_node = NULL;
        st_lookup(network->hash, network->outputs[i], (void **) &output_node);

        if (output_node != NULL && output_node->dd != NULL) {
            bdd_outputs[i] = output_node->dd;
        } else {
            printf("Error: Output node BDD for index %d not found or not built.\n", i);
            free(bdd_outputs);
            return;
        }
    }

    unsigned int total_shared_nodes = Cudd_SharingSize(bdd_outputs, noutputs);
    printf("Total number of distinct nodes across all outputs (with sharing): %u\n", total_shared_nodes);

    free(bdd_outputs);
}


void build_BDD(DdManager *dd, BnetNetwork *network) {
    BnetNode *node = network->nodes;
    while (node != NULL) {
        if (Bnet_BuildNodeBDD(dd, node, network->hash, BNET_GLOBAL_DD, 0) == 0) {
            fprintf(stderr, "Error building BDD for node: %s\n", node->name);
        } else {
            printf("Built BDD for node: %s\n", node->name);
        }
        node = node->next;
    }
}

int main(int argc, char *argv[]) {
    DdManager *dd;
    BnetNetwork *net;
    FILE *fp;
    int reorder_method;

    // if (argc != 4) {  // We now expect three arguments: BLIF file, DOT file, reorder method
    //     fprintf(stderr, "Usage: %s <file.blif> <dot_file> <reorder_method>\n", argv[0]);
    //     exit(1);
    // }

    if (argc != 3) {  // We now expect three arguments: BLIF file, DOT file, reorder method
        fprintf(stderr, "Usage: %s <file.blif>  <reorder_method>\n", argv[0]);
        exit(1);
    }

    reorder_method = atoi(argv[2]); // Convert reorder method string to integer

    // Initialize CUDD manager
    dd = Cudd_Init(0, 0, CUDD_UNIQUE_SLOTS, CUDD_CACHE_SLOTS, 0);

    // Open BLIF file
    fp = fopen(argv[1], "r");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
        Cudd_Quit(dd);
        exit(1);
    }

    // Read the BLIF file and build the BDD
    net = Bnet_ReadNetwork(fp, 1);
    printf("Successfully loaded network from BLIF file: %s\n", argv[1]);
    if (net == NULL) {
        fprintf(stderr, "Error reading BLIF file: %s\n", argv[1]);
        fclose(fp);
        Cudd_Quit(dd);
        return 1;
    }
    printf("the reorder method is = %d\n", reorder_method);
    // Enable dynamic reordering with the specified method
    Cudd_AutodynEnable(dd, reorder_method);


    // Cudd_AutodynDisable(dd);
    // int ordering[41] = {22, 33, 7, 6, 3, 38, 39, 2, 24, 19, 25, 20, 11, 37, 5, 1, 36, 0, 40, 4, 12, 16, 34, 35, 10, 15, 13, 8, 30, 21, 17, 31, 32, 27, 26, 29, 28, 14, 9, 18, 23};    // Build BDDs for all nodes


    build_BDD(dd, net);
    // Cudd_ShuffleHeap(dd, ordering);

    
    calculate_dag_sizes(dd, net);

    calculate_total_sharing_size(dd, net);

    // Cudd_PrintInfo(dd, stdout);

    // Dump all output BDDs to a single DOT file
    // dump_all_adds_to_dot(dd, net, argv[2]);

    print_bdd_variable_order(dd);
    // Cleanup
    Bnet_FreeNetwork(net);
    Cudd_Quit(dd);

    return 0;
}
