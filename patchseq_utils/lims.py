import os
import warnings
import pandas as pd
from allensdk.internal.core.lims_utilities import query

def get_cells_df(project_id=None, project_code=None, cells_list=None, has_peri_model=False,
                    has_reconstruction=False):
    """Get info for cell specimens in LIMS with ephys data
    Select by project (code or id) or list of cell specimen IDs
    
    Keyword Arguments:
        project_id -- numerical project ID
        project_code -- project code (e.g. hIVSCC-MET)
        cells_list {list} -- list of cell specimen IDs (as int or str)
    
    Returns:
        DataFrame -- cell specimen info, indexed by specimen ID
    """
    with open(os.path.join(os.path.dirname(__file__), 'cells.sql'), 'r') as sqlfile:
        sql = sqlfile.read()
    where = []
    if project_id:
        where.append("sp.project_id = {}".format(project_id))
    if project_code:
        where.append("projects.code = '{}'".format(project_code))
    if cells_list is not None:
        where.append("sp.id IN ({})".format(", ".join([str(cell) for cell in cells_list])))
    if has_peri_model:
        template = model_template_dict['peri']
        where.append("nm.neuronal_model_template_id = {}".format(template))
    if has_reconstruction:
        where.append("nr.filename IS NOT NULL")
    if len(where) > 0:
        sql += f" WHERE {' AND '.join(where)}"
    cells_df = df_from_query(sql).set_index('id')
    return cells_df

def get_ephys_features(cells_list):
    sql = "SELECT * from ephys_features ef"
    if cells_list is not None:
        sql += " WHERE ef.specimen_id IN ({})".format(", ".join([str(cell) for cell in cells_list]))
    return df_from_query(sql)

def get_ephys_qc(cells_list):
    sql = "SELECT sp.id as specimen_id, err.* FROM specimens sp JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id"
    if cells_list is not None:
        sql += " WHERE sp.id IN ({})".format(", ".join([str(cell) for cell in cells_list]))
    return df_from_query(sql)

def list_sweep_types(self):
    sql = "SELECT name from ephys_stimulus_types"
    return query(sql)

def get_sweeps_df(cell_id, sweep_type=None, description=None, passed_only=False, spiking=None, depolarizing=None):
    """Get a list of sweeps for a single cell specimen, by sweep type name
    """
    base_query = sweep_filter(cell_id, sweep_type, description, passed_only, spiking, depolarizing)
    sql = "SELECT * " + base_query
    return df_from_query(sql)

def get_sweeps(cell_id, sweep_type=None, description=None, passed_only=False, spiking=None, depolarizing=None):
    """Get a list of sweeps for a single cell specimen, by sweep type name
    """
    base_query = sweep_filter(cell_id, sweep_type, description, passed_only, spiking, depolarizing)
    sql = "SELECT sw.sweep_number " + base_query
    return query(sql)

def sweep_filter(cell_id, sweep_type=None, description=None, passed_only=False, spiking=None, depolarizing=None):
    sql = """
        FROM ephys_sweeps sw
        JOIN ephys_stimuli stim ON stim.id = sw.ephys_stimulus_id
        JOIN ephys_stimulus_types stype ON stype.id = stim.ephys_stimulus_type_id
        WHERE sw.specimen_id = {}
        """.format(cell_id)
    if sweep_type:
        sql += " AND stype.name LIKE '%%{}%%'".format(sweep_type)
    if description:
        sql += " AND stim.description LIKE '%%{}%%'".format(description)
    if passed_only:
        sql += " AND sw.workflow_state LIKE '%%passed'"
    if spiking is not None:
        sql += " AND sw.num_spikes {}".format("> 0" if spiking else "= 0")
    if depolarizing is not None:
        sql += " AND sw.stimulus_amplitude {}".format("> 0" if depolarizing else "< 0")
    return sql
    # cond = []
    # if cell_id:
    #     cond.append(f"sw.specimen_id = {t(cell_id)}")
    # if sweep_type:
    #     cond.append("stype.name LIKE '%%{}%%'".format(sweep_type)
    # if description:
    #     cond.append("stim.description LIKE '%%{}%%'".format(description)
    # if passed_only:
    #     cond.append("sw.workflow_state LIKE '%%passed'"
    # if spiking is not None:
    #     cond.append("sw.num_spikes {}".format("> 0" if spiking else "= 0")
    # if depolarizing is not None:
    #     cond.append("sw.stimulus_amplitude {}".format("> 0" if depolarizing else "< 0")
    # if len(cond) > 0:
    #     cond_str = " AND ".join(cond)
    #     sql += f"WHERE {cond_str}"
    # return sql
def count_sweeps(cell_id, distinct_amp=False, **sweep_filter_args):
    """Count sweeps for a cell specimen matching specified filter
    """
    base_query = sweep_filter(cell_id, **sweep_filter_args)
    sql = "SELECT COUNT(DISTINCT {on}) " + base_query
    distinct_col = "sw.stimulus_amplitude" if distinct_amp else "sw.id"
    return single_result_query(sql.format(on=distinct_col))

def get_sweeps_project(project_code, state=None, sweep_type=None):
    sql = """
        SELECT sw.*, 
        stype.name, stim.description
        FROM ephys_sweeps sw
        JOIN ephys_stimuli stim ON stim.id = sw.ephys_stimulus_id
        JOIN ephys_stimulus_types stype ON stype.id = stim.ephys_stimulus_type_id
        JOIN specimens sp ON sw.specimen_id = sp.id
        JOIN projects ON projects.id = sp.project_id
        WHERE projects.code = '{}'
        """.format(project_code)
    if state:
        sql += " AND sw.workflow_state = '{}'".format(state)
    if sweep_type:
        sql += " AND stype.name LIKE '%%{}%%'".format(sweep_type)
    return df_from_query(sql)
        
def get_sweep_info(cell_id, **sweep_filter_args):
    """Get a table of information for all sweeps of a given cell specimen
    """
    base_query = sweep_filter(cell_id, **sweep_filter_args)
    sql = """SELECT sw.*, 
        stype.name, stim.description
        """ + base_query
    return df_from_query(sql)

def get_nwb_path(cell_id, version='v1', get_sdk_version=False):
    """Get a network path for an NWB file from the LIMS database

    Args:
        cell_id (int or str): cell id
        version (str, optional): 'v1' or 'v2'. Defaults to 'v1'.
        get_sdk_version (bool, optional): Get the release version with spike times
        and other polish. Defaults to False.

    Returns:
        str: network path (unix formatted)
    """        
    sql = """
        SELECT nwb.storage_directory || nwb.filename AS nwb_path
        FROM specimens sp
        JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
        JOIN well_known_files nwb ON nwb.attachable_id = err.id
        JOIN well_known_file_types ftype ON nwb.well_known_file_type_id = ftype.id
        WHERE sp.id = {id}
        AND nwb.attachable_type = 'EphysRoiResult'
        """
    if version=='v1':
        sql += "AND ftype.name = 'NWB'"
    elif version=='v2':
        sql += "AND ftype.name = 'EphysNWB2'"
    if get_sdk_version:
        sql = sql[:-1] + "Download'"
    return single_result_query(sql.format(id=cell_id))

def get_lims_layers(cell_id):
    query = f"""
        select distinct
            st.acronym as name,
            polygon.path as path,
            polygon.id as polygon_id
        from specimens sp
        join specimens spp on spp.id = sp.parent_id
        join image_series imser on imser.specimen_id = spp.id
        join sub_images si on si.image_series_id = imser.id
        join images im on im.id = si.image_id
        join treatments tm on tm.id = im.treatment_id
        join avg_graphic_objects layer on layer.sub_image_id = si.id
        join avg_group_labels label on label.id = layer.group_label_id
        join avg_graphic_objects polygon on polygon.parent_id = layer.id
        join structures st on st.id = polygon.cortex_layer_id
        where 
            sp.id = {cell_id}
            and label.name in ('Cortical Layers')
            and tm.name = 'Biocytin'
        """
    return query(query)

def get_donor_info_by_cell(cell_id=None, cell_name=None):
    sql = """
    SELECT
    donors.id, 
    donors.full_genotype,
    genders.name AS sex, 
    ages.name AS age, 
    organisms.name AS species
    FROM specimens
    JOIN donors ON donors.id = sp.donor_id
    LEFT JOIN genders ON donors.gender_id = genders.id
    LEFT JOIN organisms ON donors.organism_id = organisms.id
    """
    if cell_name is not None:
        sql += f" WHERE sp.name = {cell_name}"
    elif cell_id is not None:
        sql += f" WHERE sp.id = {cell_id}"
    return single_result_query(sql)

def get_donor_info(donors):
    sql = f"""
    with medical_conditions as
        (
        select DISTINCT ON (x.donor_id)
        x.donor_id, mc.name
        from donor_medical_conditions x
        join medical_conditions mc on x.medical_condition_id = mc.id
        )
    SELECT
    donors.name,
    donors.id, 
    donors.full_genotype,
    genders.name AS sex, 
    ages.name AS age, 
    organisms.name AS species,
    medical_conditions.name AS medical_condition
    FROM donors
    LEFT JOIN genders ON donors.gender_id = genders.id
    LEFT JOIN ages ON donors.age_id = ages.id
    LEFT JOIN organisms ON donors.organism_id = organisms.id
    LEFT JOIN medical_conditions ON donors.id = medical_conditions.donor_id
    WHERE donors.name IN ({", ".join([f"'{x}'" for x in donors])})
    """
    return df_from_query(sql)
        
def get_swc_path(cell_id, manual_only=True):
    sql = """
        SELECT f.storage_directory || f.filename AS nwb_path FROM 
        neuron_reconstructions n JOIN well_known_files f ON n.id = f.attachable_id 
        WHERE n.specimen_id = {id} AND f.well_known_file_type_id = 303941301
        AND NOT n.superseded
        """
    # Not sure whether this manual flag is needed, found it in Nathan's code...
    if manual_only:
        sql += "AND n.manual"
    return single_result_query(sql.format(id=cell_id))

def get_sim_nwb_path(cell_id, model_type='peri', passed_only=True):
    template = model_template_dict[model_type]
    sql = """
        SELECT nwb.storage_directory || nwb.filename AS nwb_path
        FROM neuronal_models nm
        JOIN neuronal_model_runs runs ON runs.neuronal_model_id = nm.id
        JOIN well_known_files nwb ON nwb.attachable_id = runs.id
        JOIN well_known_file_types ftype ON nwb.well_known_file_type_id = ftype.id
        WHERE nm.specimen_id = {id}
        AND nm.neuronal_model_template_id = {template}
        AND nwb.attachable_type = 'NeuronalModelRun'
        """
    if passed_only:
        sql += "AND runs.workflow_state = 'passed'"
    return single_result_query(sql.format(id=cell_id, template=template))

def get_model_path(cell_id, model_type='peri'):
    template = model_template_dict[model_type]
    sql = """
        SELECT file.storage_directory || file.filename AS nwb_path
        FROM neuronal_models nm
        JOIN well_known_files file ON file.attachable_id = nm.id
        JOIN well_known_file_types ftype ON file.well_known_file_type_id = ftype.id
        WHERE nm.specimen_id = {id}
        AND nm.neuronal_model_template_id = {template}
        """
    return single_result_query(sql.format(id=cell_id, template=template))

def single_result_query(sql):
    results = query(sql)
    if len(results)>1:
        warnings.warn("Multiple results found in LIMS, expected single.")
    if len(results)==0:
        warnings.warn("No results found in LIMS.")
        return None
    return list(results[0].values())[0]

def df_from_query(sql):
    return pd.DataFrame.from_records(query(sql))

model_template_dict = {
        'active': 491455321,
        'lif4': 471355161,
        'lif5': 395310498,
        'lif2': 395310479,
        'lif3': 395310475,
        'lif1': 395310469,
        'peri': 329230710
        }