"""
Created on Mon Nov 25 14:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Helper routines for working with SmartSheets

"""

from collections import defaultdict

import ast
import pandas as pd
import smartsheet


class SmartSheetClient:
    def __init__(self, access_token, sheet_name, is_workspace_sheet=False):
        # Instance attributes
        self.client = smartsheet.Smartsheet(access_token)
        self.sheet_name = sheet_name

        # Open sheet
        if is_workspace_sheet:
            self.sheet_id = self.find_workspace_sheet_id()
        else:
            self.sheet_id = self.find_sheet_id()
        self.sheet = self.client.Sheets.get_sheet(self.sheet_id)
        print(self.sheet)

        # Lookups
        self.column_name_to_id = {c.title: c.id for c in self.sheet.columns}

    # --- Lookup Routines ---
    def find_workspace_sheet_id(self):
        for ws in self.client.Workspaces.list_workspaces().data:
            workspace = self.client.Workspaces.get_workspace(ws.id)
            for sheet in workspace.sheets:
                if sheet.name == self.sheet_name:
                    return sheet.id
        raise Exception(f"Sheet Not Found - sheet_name={self.sheet_name}")

    def find_sheet_id(self):
        response = self.client.Sheets.list_sheets()
        for sheet in response.data:
            if sheet.name == self.sheet_name:
                return sheet.id
        raise Exception(f"Sheet Not Found - sheet_name={self.sheet_name}")

    def find_row_id(self, keyword):
        for row in self.sheet.rows:
            for cell in row.cells:
                if cell.display_value == keyword:
                    return row.id
        raise Exception(f"Row Not Found - keyword={keyword}")

    # --- Getters ---
    def get_children_map(self):
        children_map = defaultdict(list)
        idx_lookup = {row.id: idx for idx, row in enumerate(self.sheet.rows)}
        for row in self.sheet.rows:
            if row.parent_id:
                parent_idx = idx_lookup[row.parent_id]
                child_idx = idx_lookup[row.id]
                children_map[parent_idx].append(child_idx)
        return children_map

    def get_rows_in_column_with(self, column_name, row_value):
        row_idxs = list()
        col_id = self.column_name_to_id[column_name]
        for idx, row in enumerate(self.sheet.rows):
            cell = next((c for c in row.cells if c.column_id == col_id), None)
            value = cell.display_value or cell.value
            if isinstance(value, str):
                if value.lower() == row_value.lower():
                    row_idxs.append(idx)
        return row_idxs

    def get_value(self, row_idx, column_name):
        row = self.sheet.rows[row_idx]
        col_id = self.column_name_to_id[column_name]
        cell = next((c for c in row.cells if c.column_id == col_id), None)
        return cell.display_value or cell.value

    # --- Miscellaneous ---
    def to_dataframe(self):
        # Extract column titles
        columns = list(self.column_name_to_id.keys())

        # Extract row data
        data = []
        for row in self.sheet.rows:
            row_data = []
            for cell in row.cells:
                val = cell.value if cell.display_value else cell.display_value
                row_data.append(val)
            data.append(row_data)
        return pd.DataFrame(data, columns=columns)

    def update_rows(self, updated_row):
        self.client.Sheets.update_rows(self.sheet_id, [updated_row])


# --- ExaSPIM Merge Locations Utils ---
def extract_merge_sites(smartsheet_client):
    children_map = smartsheet_client.get_children_map()
    merge_site_dfs = list()
    n_merge_sites, n_reviewed_sites = 0, 0
    for parent_idx, child_idxs in children_map.items():
        # Extract information
        sample_name = smartsheet_client.get_value(parent_idx, "Sample")
        brain_id, segmentation_id = sample_name.split("_", 1)
        sites, n = find_confirmed_merge_sites(smartsheet_client, child_idxs)

        # Compile results
        if len(sites["xyz"]) > 0:
            results = {
                "brain_id": len(sites["xyz"]) * [brain_id],
                "segmentation_id": len(sites["xyz"]) * [segmentation_id]
            }
            results.update(sites)
            merge_site_dfs.append(pd.DataFrame(results))

            n_reviewed_sites += n
            n_merge_sites += len(sites["xyz"])
            success_rate = len(sites["xyz"]) / n
            print(f"{brain_id} - Success Rate:", success_rate)

    print("\nOverall Success Rate:", n_merge_sites / n_reviewed_sites)
    print("# Confirmed Merge Sites:", n_merge_sites)
    return pd.concat(merge_site_dfs, ignore_index=True)


def find_confirmed_merge_sites(smartsheet_client, idxs):
    sites = {"segment_id": [], "groundtruth_id": [], "xyz": []}
    n_reviewed_sites = 0
    for i in idxs:
        is_merge = smartsheet_client.get_value(i, "Merge Confirmation")
        is_reviewed = smartsheet_client.get_value(i, "Reviewed?")
        if is_merge and is_reviewed:
            sites["segment_id"].append(
                smartsheet_client.get_value(i, "Segmentation ID")
            )
            sites["groundtruth_id"].append(
                smartsheet_client.get_value(i, "Ground Truth ID")
            )
            sites["xyz"].append(
                read_xyz(smartsheet_client.get_value(i, "World Coordinates"))
            )
        if is_reviewed:
            n_reviewed_sites += 1
    return sites, n_reviewed_sites


def read_xyz(xyz_str):
    return ast.literal_eval(xyz_str)
