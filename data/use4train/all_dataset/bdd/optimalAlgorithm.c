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
    for (int i = 0; i < num_vars; i++) {
        // printf("get into the loop flag");
        int var_index = Cudd_ReadInvPerm(dd, i);  // Get the variable index at position i
        printf("Position %d: Variable index %d\n", i, var_index);
    }
    printf("\n");
}

void calculate_sharing(DdManager *dd, BnetNetwork *network) {
    int noutputs = network->noutputs;
    DdNode **outputBDDs = (DdNode **)malloc(noutputs * sizeof(DdNode *));
    if (outputBDDs == NULL) {
        fprintf(stderr, "Error allocating memory for output BDD array.\n");
        return;
    }

    for (int i = 0; i < noutputs; i++) {
        BnetNode *output_node = NULL;
        st_lookup(network->hash, network->outputs[i], (void **) &output_node);

        if (output_node != NULL && output_node->dd != NULL) {
            outputBDDs[i] = output_node->dd;
            Cudd_Ref(outputBDDs[i]);  // Increment reference count to protect from garbage collection
        } else {
            printf("Error: Output node BDD for index %d not found or not built.\n", i);
            outputBDDs[i] = NULL;
        }
    }

    // Compute the total number of unique nodes shared among all outputs
    int total_size = Cudd_SharingSize(outputBDDs, noutputs);
    printf("Total number of unique BDD nodes (shared among outputs): %d\n", total_size);

    // Dereference the BDDs
    for (int i = 0; i < noutputs; i++) {
        if (outputBDDs[i] != NULL) {
            Cudd_Deref(outputBDDs[i]);
        }
    }
    free(outputBDDs);
}


void calculate_total_bdd_size(DdManager *dd, BnetNetwork *network) {
    int noutputs = network->noutputs;
    DdNode **outputBDDs = (DdNode **)malloc(noutputs * sizeof(DdNode *));
    if (outputBDDs == NULL) {
        fprintf(stderr, "Error allocating memory for output BDD array.\n");
        return;
    }

    for (int i = 0; i < noutputs; i++) {
        BnetNode *output_node = NULL;
        st_lookup(network->hash, network->outputs[i], (void **) &output_node);

        if (output_node != NULL && output_node->dd != NULL) {
            outputBDDs[i] = output_node->dd;
            Cudd_Ref(outputBDDs[i]);  // Increment reference count to protect from garbage collection
        } else {
            printf("Error: Output node BDD for index %d not found or not built.\n", i);
            outputBDDs[i] = NULL;
        }
    }

    // Compute the total number of unique nodes shared among all outputs
    int total_size = Cudd_SharingSize(outputBDDs, noutputs);
    printf("Total number of unique BDD nodes (shared among outputs): %d\n", total_size);

    // Dereference the BDDs
    for (int i = 0; i < noutputs; i++) {
        if (outputBDDs[i] != NULL) {
            Cudd_Deref(outputBDDs[i]);
        }
    }
    free(outputBDDs);
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
    // int custom_order[] = {6, 0, 13, 2, 12, 14, 11, 8, 4, 16, 5, 9, 3, 15, 10, 7, 1, 17, 18, 21, 22, 19, 20, 31, 32, 48, 59, 39, 53, 49, 23, 33, 40, 24, 41, 37, 35, 25, 42, 36, 26, 43, 38, 34, 27, 28, 30, 29, 56, 55, 54, 58, 44, 45, 46, 47, 52, 51, 57, 50};
    build_BDD(dd, net);
    
    Cudd_AutodynEnable(dd, reorder_method);

    Cudd_ReduceHeap(dd, reorder_method, 1);
    
    calculate_total_bdd_size(dd, net);


    printf("read reordering times = %d", Cudd_ReadReorderings(dd));

    Cudd_PrintInfo(dd, stdout);

    print_bdd_variable_order(dd);

    Bnet_FreeNetwork(net);
    Cudd_Quit(dd);

    return 0;
}
