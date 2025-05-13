
from allensdk.internal.core.lims_utilities import query
import allensdk.core.swc as swc

CALCULATE_AXONS = True


def get_normalized_depth(spec_id):
    sql = ""
    sql += "select sl.normalized_depth from specimens cell "
    sql += "join cell_soma_locations sl on sl.specimen_id = cell.id "
    sql += "where cell.id = "

    try:
        result = query(sql + str(spec_id))
        depth = result[0][0]
    except:
        print("Error fetching normalized soma depth specimen ID %d" % spec_id)
        raise
    return depth

def make_upright_swc(spec_id, outfile):
    print("Writing upright morphology of specimen %d to %s" % (spec_id, outfile))
    nrn = make_upright_morphology(spec_id)
    try:
        nrn.write(outfile)
    except:
        print("Error writing upright morphology file '%s'" % outfile)
        raise


def make_upright_morphology(spec_id):
    ####################################################################
    # SQL queries

    aff_sql = ""
    aff_sql += "SELECT "
    aff_sql += "  a3d.tvr_00, a3d.tvr_01, a3d.tvr_02, "
    aff_sql += "  a3d.tvr_03, a3d.tvr_04, a3d.tvr_05, "
    aff_sql += "  a3d.tvr_06, a3d.tvr_07, a3d.tvr_08, "
    aff_sql += "  a3d.tvr_09, a3d.tvr_10, a3d.tvr_11 "
    aff_sql += "FROM specimens spc "
    aff_sql += "JOIN neuron_reconstructions nr ON nr.specimen_id=spc.id "
    aff_sql += "  AND nr.superseded = 'f' AND nr.manual = 't' "
    aff_sql += "JOIN alignment3ds a3d ON a3d.id=spc.alignment3d_id "
    aff_sql += "WHERE spc.id = %d;"

    path_sql = ""
    path_sql += "SELECT distinct cell.storage_directory || wkf.filename "
    path_sql += "FROM neuron_reconstructions nr "
    path_sql += "JOIN well_known_files wkf on wkf.attachable_id = nr.id "
    path_sql += "JOIN specimens cell on cell.id = nr.specimen_id "
    path_sql += "WHERE wkf.filename ilike '%_m.swc' "
    path_sql += "AND nr.superseded = false "
    path_sql += "AND wkf.filename not ilike '%marker%' "
    path_sql += "AND cell.id = "

    # grap affine transform

    try:
        result = query(aff_sql % spec_id)
    #    print(aff_sql)
    #    print(len(result))
        aff = []
        if len(result) > 0:
            for i in range(12):
    #            print(i)
    #            print(result[0][i])
                aff.append(result[0][i])
    except:
        print("Error fetching affine transform for specimen ID %d" % spec_id)
        raise
            
    # fetch swc file
    try:
        result = query(path_sql + str(spec_id))
        swc_file = result[0][0]
    except:
        print("Error fetching SWC file name for specimen ID %d" % spec_id)
        raise

    # create morphology
    try:
        # swc_file.replace('/', '\\')
        # swc_file = "\\" + swc_file
        nrn = swc.read_swc(swc_file)
    except:
        print("Error fetching SWC file '%s'" % swc_file)
        raise

    # apply transform and save file
    try:
        nrn.apply_affine(aff, scale=1)
        # print aff
    except:
        print("Error applying affine transform")
        raise
    # nrn.save(str(spec_id) + ".swc")
    return nrn

