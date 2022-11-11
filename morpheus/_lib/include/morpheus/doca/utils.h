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

#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <doca_error.h>
#include <doca_log.h>
#include <doca_dev.h>

#define APP_EXIT(format, ...)					\
	do {							\
		DOCA_LOG_ERR(format "\n", ##__VA_ARGS__);	\
		exit(1);					\
	} while (0)

#if __cplusplus
extern "C" {
#endif

typedef doca_error_t (*caps_check)(struct doca_devinfo *, uint32_t *);

doca_error_t sdk_version_callback(void *doca_config, void *param);

/* parse string pci address into bdf struct */
doca_error_t parse_pci_addr(char const *pci_addr, struct doca_pci_bdf *out_bdf);

/* read the entire content of a file into a buffer */
doca_error_t read_file(char const *path, char **out_bytes, size_t *out_bytes_len);

#if __cplusplus
}
#endif

#endif /* COMMON_UTILS_H_ */
