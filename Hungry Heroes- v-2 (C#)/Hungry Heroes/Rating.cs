using Hungry_Heroes.Interfaces;
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
    public partial class Rating : Form, IRating
    {
        PaymentSuccess paymentSuccess;
        private string resturantName;
        public Rating(PaymentSuccess paymentSuccess, string resturantName)
        {
            InitializeComponent();
            this.paymentSuccess = paymentSuccess;
            paymentSuccess.ratingcheck = 1;
            this.resturantName = resturantName;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void brate1_Click(object sender, EventArgs e)
        {
            //for rating 1
            //Database Start
            SqlConnection con = databaseConnection();
            //getting last bill id
            SqlCommand cmd = new SqlCommand("SELECT NEXT VALUE FOR BILL_ID_SEQUENCE", con);
            string sid = cmd.ExecuteScalar().ToString();
            int id = int.Parse(sid);
            id--;

            //updating bill
            cmd = new SqlCommand("UPDATE BILL SET RATING=@rating WHERE BILL_ID=@id", con);
            cmd.Parameters.AddWithValue("@rating", 1);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();
            //updating resturant rating
            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET TOTAL_RATING = TOTAL_RATING + @rate, TOTAL_ORDER = TOTAL_ORDER + @order WHERE RESTURANT_NAME = @name", con);
            cmd.Parameters.AddWithValue("@rate", 1);
            cmd.Parameters.AddWithValue("@order", 1);
            cmd.Parameters.AddWithValue("@name", resturantName);
            cmd.ExecuteNonQuery();
            con.Close();
            //database end

            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
            paymentSuccess.ratingdone = 1;
        }

        public void rate2_Click(object sender, EventArgs e)
        {
            //for rating 2
            //Database Start
            SqlConnection con = databaseConnection();
            //getting last bill id
            SqlCommand cmd = new SqlCommand("SELECT NEXT VALUE FOR BILL_ID_SEQUENCE", con);
            string sid = cmd.ExecuteScalar().ToString();
            int id = int.Parse(sid);
            id--;

            //updating bill
            cmd = new SqlCommand("UPDATE BILL SET RATING=@rating WHERE BILL_ID=@id", con);
            cmd.Parameters.AddWithValue("@rating", 2);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();
            //updating resturant rating
            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET TOTAL_RATING = TOTAL_RATING + @rate, TOTAL_ORDER = TOTAL_ORDER + @order WHERE RESTURANT_NAME = @name",con);
            cmd.Parameters.AddWithValue("@rate", 2);
            cmd.Parameters.AddWithValue("@order", 1);
            cmd.Parameters.AddWithValue("@name", resturantName);
            cmd.ExecuteNonQuery();
            con.Close();
            //database end
            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
            paymentSuccess.ratingdone = 1;
        }

        public void button2_Click(object sender, EventArgs e)
        {
            //for rating 3
            //Database Start
            SqlConnection con = databaseConnection();
            //getting last bill id
            SqlCommand cmd = new SqlCommand("SELECT NEXT VALUE FOR BILL_ID_SEQUENCE", con);
            string sid = cmd.ExecuteScalar().ToString();
            int id = int.Parse(sid);
            id--;

            //updating bill
            cmd = new SqlCommand("UPDATE BILL SET RATING=@rating WHERE BILL_ID=@id", con);
            cmd.Parameters.AddWithValue("@rating", 3);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();
            //updating resturant rating
            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET TOTAL_RATING = TOTAL_RATING + @rate, TOTAL_ORDER = TOTAL_ORDER + @order WHERE RESTURANT_NAME = @name", con);
            cmd.Parameters.AddWithValue("@rate", 3);
            cmd.Parameters.AddWithValue("@order", 1);
            cmd.Parameters.AddWithValue("@name", resturantName);
            cmd.ExecuteNonQuery();
            con.Close();
            //database end
            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
            paymentSuccess.ratingdone = 1;
        }

        public void button3_Click(object sender, EventArgs e)
        {
            //for rating 4
            //Database Start
            SqlConnection con = databaseConnection();
            //getting last bill id
            SqlCommand cmd = new SqlCommand("SELECT NEXT VALUE FOR BILL_ID_SEQUENCE", con);
            string sid = cmd.ExecuteScalar().ToString();
            int id = int.Parse(sid);
            id--;

            //updating bill
            cmd = new SqlCommand("UPDATE BILL SET RATING=@rating WHERE BILL_ID=@id", con);
            cmd.Parameters.AddWithValue("@rating", 4);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();
            //updating resturant rating
            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET TOTAL_RATING = TOTAL_RATING + @rate, TOTAL_ORDER = TOTAL_ORDER + @order WHERE RESTURANT_NAME = @name",con);
            cmd.Parameters.AddWithValue("@rate", 4);
            cmd.Parameters.AddWithValue("@order", 1);
            cmd.Parameters.AddWithValue("@name", resturantName);
            cmd.ExecuteNonQuery();
            con.Close();
            //database end
            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
            paymentSuccess.ratingdone = 1;
        }

        public void button4_Click(object sender, EventArgs e)
        {
            //for rating 5
            //Database Start
            SqlConnection con = databaseConnection();
            //getting last bill id
            SqlCommand cmd = new SqlCommand("SELECT NEXT VALUE FOR BILL_ID_SEQUENCE", con);
            int id = int.Parse(cmd.ExecuteScalar().ToString());
            id--;

            //updating bill
            cmd = new SqlCommand("UPDATE BILL SET RATING=@rating WHERE BILL_ID=@id", con);
            cmd.Parameters.AddWithValue("@rating", 5);
            cmd.Parameters.AddWithValue("@id", id);
            cmd.ExecuteNonQuery();


            //updating resturant rating
            cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET TOTAL_RATING = TOTAL_RATING + @rate, TOTAL_ORDER = TOTAL_ORDER + @order WHERE RESTURANT_NAME = @name",con);
            cmd.Parameters.AddWithValue("@rate", 5);
            cmd.Parameters.AddWithValue("@order", 1);
            cmd.Parameters.AddWithValue("@name", resturantName);
            cmd.ExecuteNonQuery();
            con.Close();
            //database end
            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
            paymentSuccess.ratingdone = 1;
        }

        public void bprevious_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            paymentSuccess.ratingcheck = 0;
        }
    }
}
