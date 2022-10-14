/*
 * Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <arpa/inet.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdnoreturn.h>

#include <doca_version.h>
#include <doca_log.h>

#include "morpheus/doca/utils.h"

#define GET_BYTE(V, N)	((uint8_t)((V) >> ((N) * 8) & 0xFF))
#define SET_BYTE(V, N)	(((V) & 0xFF)  << ((N) * 8))

DOCA_LOG_REGISTER(UTILS);

noreturn doca_error_t
sdk_version_callback(void *param, void *doca_config)
{
	printf("DOCA SDK     Version (Compilation): %s\n", doca_version());
	printf("DOCA Runtime Version (Runtime):     %s\n", doca_version_runtime());
	/* We assume that when printing DOCA's versions there is no need to continue the program's execution */
	exit(EXIT_SUCCESS);
}

doca_error_t
parse_pci_addr(char const *pci_addr, struct doca_pci_bdf *out_bdf)
{
	unsigned int bus_bitmask = 0xFFFFFF00;
	unsigned int dev_bitmask = 0xFFFFFFE0;
	unsigned int func_bitmask = 0xFFFFFFF8;
	uint32_t tmpu;
	char tmps[4];

	if (pci_addr == NULL || strlen(pci_addr) != 7 || pci_addr[2] != ':' || pci_addr[5] != '.')
		return DOCA_ERROR_INVALID_VALUE;

	tmps[0] = pci_addr[0];
	tmps[1] = pci_addr[1];
	tmps[2] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & bus_bitmask) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	out_bdf->bus = tmpu;

	tmps[0] = pci_addr[3];
	tmps[1] = pci_addr[4];
	tmps[2] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & dev_bitmask) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	out_bdf->device = tmpu;

	tmps[0] = pci_addr[6];
	tmps[1] = '\0';
	tmpu = strtoul(tmps, NULL, 16);
	if ((tmpu & func_bitmask) != 0)
		return DOCA_ERROR_INVALID_VALUE;
	out_bdf->function = tmpu;

	return DOCA_SUCCESS;
}

doca_error_t
read_file(char const *path, char **out_bytes, size_t *out_bytes_len)
{
	FILE *file;
	char *bytes;

	file = fopen(path, "rb");
	if (file == NULL)
		return DOCA_ERROR_NOT_FOUND;

	if (fseek(file, 0, SEEK_END) != 0) {
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	long const nb_file_bytes = ftell(file);

	if (nb_file_bytes == -1) {
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	if (nb_file_bytes == 0) {
		fclose(file);
		return DOCA_ERROR_INVALID_VALUE;
	}

	bytes = malloc(nb_file_bytes);
	if (bytes == NULL) {
		fclose(file);
		return DOCA_ERROR_NO_MEMORY;
	}

	if (fseek(file, 0, SEEK_SET) != 0) {
		free(bytes);
		fclose(file);
		return DOCA_ERROR_IO_FAILED;
	}

	size_t const read_byte_count = fread(bytes, 1, nb_file_bytes, file);

	fclose(file);

	if (read_byte_count != nb_file_bytes) {
		free(bytes);
		return DOCA_ERROR_IO_FAILED;
	}

	*out_bytes = bytes;
	*out_bytes_len = read_byte_count;

	return DOCA_SUCCESS;
}

uint64_t
ntohq(uint64_t value)
{
	/* If we are in a Big-Endian architecture, we don't need to do anything */
	if (ntohl(1) == 1)
		return value;

	/* Swap the 8 bytes of our value */
	value = SET_BYTE((uint64_t)GET_BYTE(value, 0), 7) | SET_BYTE((uint64_t)GET_BYTE(value, 1), 6) |
		SET_BYTE((uint64_t)GET_BYTE(value, 2), 5) | SET_BYTE((uint64_t)GET_BYTE(value, 3), 4) |
		SET_BYTE((uint64_t)GET_BYTE(value, 4), 3) | SET_BYTE((uint64_t)GET_BYTE(value, 5), 2) |
		SET_BYTE((uint64_t)GET_BYTE(value, 6), 1) | SET_BYTE((uint64_t)GET_BYTE(value, 7), 0);

	return value;
}
