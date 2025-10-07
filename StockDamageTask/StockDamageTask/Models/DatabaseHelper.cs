using System;
using System.Collections.Generic;
using System.Configuration;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Security.Policy;
using System.Web;

namespace StockDamageTask.Models
{
    public class DatabaseHelper
    {
        // connection is only realonly so no one can manipulate it
        private readonly SqlConnection _connection;

        // in constructor it is getting connection string for db defined by me in web.config file
        public DatabaseHelper()
        {
            string connectionString = ConfigurationManager.ConnectionStrings["StockDamageTaskDatabaseConnection"].ConnectionString;
            _connection = new SqlConnection(connectionString);
        }

        // it will open the connection of the database if it is closed
        public SqlConnection GetConnection()
        {
            if (_connection.State == ConnectionState.Closed)
            {
                _connection.Open();
            }
            return _connection;
        }

        // it will close connection from database if the connection is open
        public void CloseConnection()
        {
            if (_connection.State == ConnectionState.Open)
            {
                _connection.Close();
            }
        }

        // it will execute query which can retun multiple value (row)
        // open and close connection is establised inside so no need manual close connection
        // this fuction will be used for select statement
        public DataTable ExecuteQuery(string query)
        {
            DataTable dt = new DataTable();
            using (SqlCommand cmd = new SqlCommand(query, GetConnection()))
            {
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                da.Fill(dt);
            }
            CloseConnection();
            return dt;
        }

        // it will execut query which will not retun value. it will only retun the affected no
        // of rows in the database. it can be used to add data to database or update data or delete
        public int ExecuteNonQuery(string query, SqlParameter[] parameters)
        {
            using (SqlCommand cmd = new SqlCommand(query, GetConnection()))
            {
                if (parameters != null)
                {
                    cmd.Parameters.AddRange(parameters);
                }
                int rows = cmd.ExecuteNonQuery();
                CloseConnection();
                return rows;
            }
        }

        // it can be used for sum, count, and other type of gorup fucntion which
        // typically give single value
        public object ExecuteScalar(string query, SqlParameter[] parameters = null)
        {
            using (SqlCommand cmd = new SqlCommand(query, GetConnection()))
            {
                if (parameters != null)
                {
                    cmd.Parameters.AddRange(parameters);
                }
                object result = cmd.ExecuteScalar();
                CloseConnection();
                return result;
            }
        }

        // this can be used to call procedure.
        // it will receive procedure name, its parameter and command type 
        // in sql server a procedure is defined to handle create update delete logic
        public DataTable ExecuteQueryWithParameters(string storedProcName, SqlParameter[] parameters, CommandType cmdType = CommandType.Text)
        {
            DataTable dt = new DataTable();
            using (SqlCommand cmd = new SqlCommand(storedProcName, GetConnection()))
            {
                cmd.CommandType = cmdType;
                if (parameters != null)
                {
                    cmd.Parameters.AddRange(parameters);
                }
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                da.Fill(dt);
            }
            CloseConnection();
            return dt;
        }

        // it will dispose the connection for the database
        public void Dispose()
        {
            CloseConnection();
            _connection.Dispose();
        }
    }

    // This code is done by Faysal Ahmmed - mail:faysalahmmed4200 @gmail.com
}