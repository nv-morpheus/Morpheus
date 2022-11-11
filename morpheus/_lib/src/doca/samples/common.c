/*
 * Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
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

#include <morpheus/doca/samples/common.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_mmap.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

DOCA_LOG_REGISTER(COMMON);

#define MAX_REP_PROPERTY_LEN 128

doca_error_t
open_doca_device_with_pci(const struct doca_pci_bdf *value, jobs_check func, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	struct doca_pci_bdf buf = {};
	int res;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	res = doca_devinfo_list_create(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_get_pci_addr(dev_list[i], &buf);
		if (res == DOCA_SUCCESS && buf.raw == value->raw) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_list_destroy(dev_list);
				return res;
			}
		}
	}

	DOCA_LOG_ERR("Matching device not found.");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_list_destroy(dev_list);
	return res;
}

doca_error_t
open_doca_device_with_ibdev_name(const uint8_t *value, size_t val_size, jobs_check func,
					 struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	char buf[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	char val_copy[DOCA_DEVINFO_IBDEV_NAME_SIZE] = {};
	int res;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	/* Setup */
	if (val_size > DOCA_DEVINFO_IBDEV_NAME_SIZE) {
		DOCA_LOG_ERR("Value size too large. Failed to locate device.");
		return DOCA_ERROR_INVALID_VALUE;
	}
	memcpy(val_copy, value, val_size);

	res = doca_devinfo_list_create(&dev_list, &nb_devs);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", res);
		return res;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		res = doca_devinfo_get_ibdev_name(dev_list[i], buf, DOCA_DEVINFO_IBDEV_NAME_SIZE);
		if (res == DOCA_SUCCESS && memcmp(buf, val_copy, DOCA_DEVINFO_IBDEV_NAME_SIZE) == 0) {
			/* If any special capabilities are needed */
			if (func != NULL && func(dev_list[i]) != DOCA_SUCCESS)
				continue;

			/* if device can be opened */
			res = doca_dev_open(dev_list[i], retval);
			if (res == DOCA_SUCCESS) {
				doca_devinfo_list_destroy(dev_list);
				return res;
			}
		}
	}

	DOCA_LOG_ERR("Matching device not found.");
	res = DOCA_ERROR_NOT_FOUND;

	doca_devinfo_list_destroy(dev_list);
	return res;
}

doca_error_t
open_doca_device_with_capabilities(jobs_check func, struct doca_dev **retval)
{
	struct doca_devinfo **dev_list;
	uint32_t nb_devs;
	doca_error_t result;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	result = doca_devinfo_list_create(&dev_list, &nb_devs);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to load doca devices list. Doca_error value: %d", result);
		return result;
	}

	/* Search */
	for (i = 0; i < nb_devs; i++) {
		/* If any special capabilities are needed */
		if (func(dev_list[i]) != DOCA_SUCCESS)
			continue;

		/* If device can be opened */
		if (doca_dev_open(dev_list[i], retval) == DOCA_SUCCESS) {
			doca_devinfo_list_destroy(dev_list);
			return DOCA_SUCCESS;
		}
	}

	DOCA_LOG_ERR("Matching device not found.");
	doca_devinfo_list_destroy(dev_list);
	return DOCA_ERROR_NOT_FOUND;
}

doca_error_t
open_doca_device_rep_with_vuid(struct doca_dev *local, enum doca_dev_rep_filter filter, const uint8_t *value,
				       size_t val_size, struct doca_dev_rep **retval)
{
	uint32_t nb_rdevs = 0;
	struct doca_devinfo_rep **rep_dev_list = NULL;
	char val_copy[DOCA_DEVINFO_REP_VUID_SIZE] = {};
	char buf[DOCA_DEVINFO_REP_VUID_SIZE] = {};
	doca_error_t result;
	size_t i;

	/* Set default return value */
	*retval = NULL;

	/* Setup */
	if (val_size > DOCA_DEVINFO_REP_VUID_SIZE) {
		DOCA_LOG_ERR("Value size too large. Ignored.");
		return DOCA_ERROR_INVALID_VALUE;
	}
	memcpy(val_copy, value, val_size);

	/* Search */
	result = doca_devinfo_rep_list_create(local, filter, &rep_dev_list, &nb_rdevs);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to create devinfo representor list. Representor devices are available only on DPU, do not run on Host.");
		return DOCA_ERROR_INVALID_VALUE;
	}

	for (i = 0; i < nb_rdevs; i++) {
		result = doca_devinfo_rep_get_vuid(rep_dev_list[i], buf, DOCA_DEVINFO_REP_VUID_SIZE);
		if (result == DOCA_SUCCESS && memcmp(buf, val_copy, DOCA_DEVINFO_REP_VUID_SIZE) == 0 &&
		    doca_dev_rep_open(rep_dev_list[i], retval) == DOCA_SUCCESS) {
			doca_devinfo_rep_list_destroy(rep_dev_list);
			return DOCA_SUCCESS;
		}
	}

	DOCA_LOG_ERR("Matching device not found.");
	doca_devinfo_rep_list_destroy(rep_dev_list);
	return DOCA_ERROR_NOT_FOUND;
}

doca_error_t
open_doca_device_rep_with_pci(struct doca_dev *local, enum doca_dev_rep_filter filter, struct doca_pci_bdf *pci_bdf,
			      struct doca_dev_rep **retval)
{
	uint32_t nb_rdevs = 0;
	struct doca_devinfo_rep **rep_dev_list = NULL;
	struct doca_pci_bdf queried_pci_bdf;
	doca_error_t result;
	size_t i;

	*retval = NULL;

	/* Search */
	result = doca_devinfo_rep_list_create(local, filter, &rep_dev_list, &nb_rdevs);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR(
			"Failed to create devinfo representors list. Representor devices are available only on DPU, do not run on Host.");
		return DOCA_ERROR_INVALID_VALUE;
	}

	for (i = 0; i < nb_rdevs; i++) {
		result = doca_devinfo_rep_get_pci_addr(rep_dev_list[i], &queried_pci_bdf);
		if (result == DOCA_SUCCESS && queried_pci_bdf.raw == pci_bdf->raw &&
		    doca_dev_rep_open(rep_dev_list[i], retval) == DOCA_SUCCESS) {
			doca_devinfo_rep_list_destroy(rep_dev_list);
			return DOCA_SUCCESS;
		}
	}

	DOCA_LOG_ERR("Matching device not found.");
	doca_devinfo_rep_list_destroy(rep_dev_list);
	return DOCA_ERROR_NOT_FOUND;
}

doca_error_t
init_core_objects(struct program_core_objects *state, uint32_t extensions, uint32_t workq_depth, uint32_t max_chunks)
{
	doca_error_t res;
	struct doca_workq *workq;

	res = doca_mmap_create(NULL, &state->mmap);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create mmap: %s", doca_get_error_string(res));
		return res;
	}

	res = doca_buf_inventory_create(NULL, max_chunks, extensions, &state->buf_inv);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create buffer inventory: %s", doca_get_error_string(res));
		return res;
	}

	res = doca_mmap_set_max_num_chunks(state->mmap, max_chunks);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set memory map nb chunks: %s", doca_get_error_string(res));
		return res;
	}

	res = doca_mmap_start(state->mmap);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start memory map: %s", doca_get_error_string(res));
		doca_mmap_destroy(state->mmap);
		state->mmap = NULL;
		return res;
	}

	res = doca_mmap_dev_add(state->mmap, state->dev);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to add device to mmap: %s", doca_get_error_string(res));
		doca_mmap_destroy(state->mmap);
		state->mmap = NULL;
		return res;
	}

	res = doca_buf_inventory_start(state->buf_inv);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start buffer inventory: %s", doca_get_error_string(res));
		return res;
	}

	res = doca_ctx_dev_add(state->ctx, state->dev);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to register device with lib context: %s", doca_get_error_string(res));
		state->ctx = NULL;
		return res;
	}

	res = doca_ctx_start(state->ctx);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start lib context: %s", doca_get_error_string(res));
		doca_ctx_dev_rm(state->ctx, state->dev);
		state->ctx = NULL;
		return res;
	}

	res = doca_workq_create(workq_depth, &workq);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create work queue: %s", doca_get_error_string(res));
		return res;
	}

	res = doca_ctx_workq_add(state->ctx, workq);
	if (res != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to register work queue with context: %s", doca_get_error_string(res));
		doca_workq_destroy(workq);
		state->workq = NULL;
	} else
		state->workq = workq;

	return res;
}

doca_error_t
destroy_core_objects(struct program_core_objects *state)
{
	doca_error_t tmp_result, result = DOCA_SUCCESS;

	if (state->workq != NULL) {
		tmp_result = doca_ctx_workq_rm(state->ctx, state->workq);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to remove work queue from ctx: %s", doca_get_error_string(tmp_result));
		}

		tmp_result = doca_workq_destroy(state->workq);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to destroy work queue: %s", doca_get_error_string(tmp_result));
		}
		state->workq = NULL;
	}

	if (state->buf_inv != NULL) {
		tmp_result = doca_buf_inventory_destroy(state->buf_inv);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to destroy buf inventory: %s", doca_get_error_string(tmp_result));
		}
		state->buf_inv = NULL;
	}

	if (state->mmap != NULL) {
		tmp_result = doca_mmap_dev_rm(state->mmap, state->dev);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to remove device from mmap: %s", doca_get_error_string(tmp_result));
		}

		tmp_result = doca_mmap_destroy(state->mmap);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to destroy mmap: %s", doca_get_error_string(tmp_result));
		}
		state->mmap = NULL;
	}

	if (state->ctx != NULL) {
		tmp_result = doca_ctx_stop(state->ctx);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Unable to stop context: %s", doca_get_error_string(tmp_result));
		}

		tmp_result = doca_ctx_dev_rm(state->ctx, state->dev);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to remove device from ctx: %s", doca_get_error_string(tmp_result));
		}
	}

	if (state->dev != NULL) {
		tmp_result = doca_dev_close(state->dev);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_ERROR_PROPAGATE(result, tmp_result);
			DOCA_LOG_ERR("Failed to close device: %s", doca_get_error_string(tmp_result));
		}
		state->dev = NULL;
	}

	return result;
}

char *
hex_dump(const void *data, size_t size)
{
	/*
	 * <offset>:     <Hex bytes: 1-8>        <Hex bytes: 9-16>         <Ascii>
	 * 00000000: 31 32 33 34 35 36 37 38  39 30 61 62 63 64 65 66  1234567890abcdef
	 *    8     2         8 * 3          1          8 * 3         1       16       1
	 */
	const size_t line_size = 8 + 2 + 8 * 3 + 1 + 8 * 3 + 1 + 16 + 1;
	int i, j, r, read_index;
	size_t num_lines, buffer_size;
	char *buffer, *write_head;
	unsigned char cur_char, printable;
	char ascii_line[17];
	const unsigned char *input_buffer;

	/* Allocate a dynamic buffer to hold the full result */
	num_lines = (size + 16 - 1) / 16;
	buffer_size = num_lines * line_size + 1;
	buffer = (char *)malloc(buffer_size);
	if (buffer == NULL)
		return NULL;
	write_head = buffer;
	input_buffer = data;
	read_index = 0;

	for (i = 0; i < num_lines; i++)	{
		/* Offset */
		snprintf(write_head, buffer_size, "%08X: ", i * 16);
		write_head += 8 + 2;
		buffer_size -= 8 + 2;
		/* Hex print - 2 chunks of 8 bytes */
		for (r = 0; r < 2 ; r++) {
			for (j = 0; j < 8; j++) {
				/* If there is content to print */
				if (read_index < size) {
					cur_char = input_buffer[read_index++];
					snprintf(write_head, buffer_size, "%02X ", cur_char);
					/* Printable chars go "as-is" */
					if (' ' <= cur_char && cur_char <= '~')
						printable = cur_char;
					/* Otherwise, use a '.' */
					else
						printable = '.';
				/* Else, just use spaces */
				} else {
					snprintf(write_head, buffer_size, "   ");
					printable = ' ';
				}
				ascii_line[r * 8 + j] = printable;
				write_head += 3;
				buffer_size -= 3;
			}
			/* Spacer between the 2 hex groups */
			snprintf(write_head, buffer_size, " ");
			write_head += 1;
			buffer_size -= 1;
		}
		/* Ascii print */
		ascii_line[16] = '\0';
		snprintf(write_head, buffer_size, "%s\n", ascii_line);
		write_head += 16 + 1;
		buffer_size -= 16 + 1;
	}
	/* No need for the last '\n' */
	write_head[-1] = '\0';
	return buffer;
}
