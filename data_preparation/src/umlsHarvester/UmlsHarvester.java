/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package umlsHarvester;

//import edu.emory.cci.aiw.umls.*;
// import mysql-connector-java-5.1.44-bin.jar
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * This script is used to get information from a local database of UMLS in MySQL
 *      To create the MySQL database of UMLS, use the MetamorphoSys system
 *      Add UMLS SN tables to use functions for semantic types, relations etc.
 * @author tasosnent
 */
public class UmlsHarvester {    
    protected UMLSDatabaseConnection conn; // The connection to the database using the UMLSQuery library
    
    /**
     *  Initialize a UmlsHarvester object
     *      Setup a connection to UMLS DB
     * @param dbURL         The URL for the database (e.g. jdbc:mysql://localhost/umls_ad)
     * @param user          The user name for database connection (e.g. root)
     * @param pass          The password for database connection (e.g. testpass1)
     */
    public UmlsHarvester(String dbURL,String user, String pass){     
        //Create connection to UMLS MySQL database
        conn = UMLSDatabaseConnection.getConnection(dbURL, user, pass);     
        
    } 
    
    public static void main(String[] args) {


    }
    

    /**
     * Get the CUI associated to the given Source Concept code
     *      Tested for : MESH
     * @param SourceID      The source concept id (e.g. M0000842)
     * @param sab           The source (SAB) name (e.g. "MSH2017_2017_02_14")
     * @return              A ConceptUID for this concept
     */     
    public String getCUIByCUID(String SourceID, String sab){
        String cui = "";
        try {
                cui = conn.SCUIcodeToUID(SourceID, sab);
            } catch (UMLSQueryException ex) {
            Logger.getLogger(UmlsHarvester.class.getName()).log(Level.SEVERE, null, ex);
        }
        return cui;
    }    
    
}
