with dendrite_type as
    (
    select sts.specimen_id, st.name
    from specimen_tags_specimens sts
    join specimen_tags st on sts.specimen_tag_id = st.id
    where st.name like 'dendrite type%%'
    )
, imaging as
    (
    select DISTINCT ON (sts.specimen_id)
    sts.specimen_id, st.name
    from specimen_tags_specimens sts
    join specimen_tags st on sts.specimen_tag_id = st.id
    where st.name like '63x%%'
    ORDER BY sts.specimen_id, st.name DESC
    )
SELECT sp.id, sp.name, sp.created_at, 
csl.normalized_depth, csl.soma_depth_um,
struct.acronym AS structure, 
hemispheres.name AS hemisphere,
donors.name AS donor_name, 
-- donors.full_genotype,
-- ages.name AS donor_age, 
dt.name AS dendrite_type,
im.name AS img_status,
layer.name AS layer,
projects.code AS project_code,
err.id AS err_id,
nwb.storage_directory || nwb.filename AS nwb_path,
nwb.ftype as nwb_type,
nr.reconstruction_tag,
nr.reconstruction_type,
nr.storage_directory || nr.filename AS swc_path
FROM specimens sp
LEFT JOIN ephys_roi_results err ON sp.ephys_roi_result_id = err.id
LEFT JOIN projects ON projects.id = sp.project_id
LEFT JOIN donors ON donors.id = sp.donor_id
LEFT JOIN ages ON donors.age_id = ages.id
LEFT JOIN structures struct ON sp.structure_id = struct.id
LEFT JOIN hemispheres ON sp.hemisphere_id = hemispheres.id
LEFT JOIN structures layer ON sp.cortex_layer_id = layer.id
LEFT JOIN cell_soma_locations csl ON csl.specimen_id = sp.id
LEFT JOIN dendrite_type dt ON sp.id = dt.specimen_id
LEFT JOIN imaging im ON sp.id = im.specimen_id
LEFT JOIN neuronal_models nm ON sp.id = nm.specimen_id
LEFT JOIN
(
    SELECT DISTINCT ON (nr.specimen_id)
    nr.specimen_id, f.storage_directory, f.filename, rt.name AS reconstruction_tag,
    nrt.name as reconstruction_type
    FROM neuron_reconstructions nr 
    JOIN well_known_files f ON nr.id = f.attachable_id 
    LEFT JOIN neuron_reconstruction_tags_neuron_reconstructions rtr ON rtr.neuron_reconstruction_id = nr.id
    LEFT JOIN neuron_reconstruction_tags rt on rtr.neuron_reconstruction_tag_id = rt.id
    LEFT JOIN reconstruction_types nrt on nr.reconstruction_type_id = nrt.id
    WHERE f.well_known_file_type_id = 303941301 --swc
    AND (rt.name != 'dendrite-only' OR rt.name IS NULL)
    AND NOT nr.superseded
    AND nr.manual
    -- should pick 63x over 100x by ordering
    ORDER BY nr.specimen_id, nr.reconstruction_type_id
) AS nr
ON sp.id = nr.specimen_id
LEFT JOIN 
(
    SELECT DISTINCT ON (nwb.attachable_id)
    nwb.storage_directory, nwb.filename, nwb.attachable_id, ftype.name as ftype FROM 
    well_known_files nwb 
    JOIN well_known_file_types ftype ON nwb.well_known_file_type_id = ftype.id
    WHERE nwb.attachable_type = 'EphysRoiResult'
    AND ftype.name IN ('NWB', 'EphysNWB2')
    ORDER BY nwb.attachable_id, ftype.name
) AS nwb
ON nwb.attachable_id = err.id