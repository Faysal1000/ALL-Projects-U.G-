using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Hungry_Heroes
{
    public partial class PreviousOrderReuse : UserControl
    {
        private string username;
        public PreviousOrderReuse(string username)
        {
            InitializeComponent();
            this.username = username;
        }
        private SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        private void PreviousOrderReuse_Load(object sender, EventArgs e)
        {
            //Database start
            SqlConnection con = databaseConnection();

            SqlCommand cmd = new SqlCommand("SELECT BILLDATE,RESTURANT_NAME,ITEM_1,ITEM_2,ITEM_3,ITEM_4,ITEM_5,DISCOUNT,TOTAL FROM BILL WHERE USERNAME = @username COLLATE Latin1_General_BIN", con);
            cmd.Parameters.AddWithValue("@username", username);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.None;
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //Database end
        }
    }
}
