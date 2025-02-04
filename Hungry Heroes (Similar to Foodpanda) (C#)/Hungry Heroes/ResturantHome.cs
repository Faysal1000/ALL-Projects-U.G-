using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Data.SqlClient;
using System.Drawing;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.ListView;
using System.Xml.Linq;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.StartPanel;
using Hungry_Heroes.Interfaces;

namespace Hungry_Heroes
{
    public partial class ResturantHome : Form, IResturantHome
    {
        private HomeReuse homeReuse;
        private ResturantAvailability resturantAvailability;
        private string resturantname;
        private string foodname;
        public ResturantHome(string resturantname)
        {
            InitializeComponent();
            homeReuse = new HomeReuse();
            resturantAvailability = new ResturantAvailability(resturantname);
            this.resturantname = resturantname;
            pHome.Controls.Add(homeReuse);
            pb1.Visible = true;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void bhome_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Home";
            //for button underline panel
            pb1.Visible = true;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;
            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;
            bclear.Visible = false;

            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            //new added for discount
            bupdatediscount.Visible = false;
            ldiscount.Visible = false;
            tbdiscount.Visible = false;
            lcurrentdiscount.Visible = false;

            //remove user control
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(resturantAvailability);
            //Add user control to pHome Panel
            pHome.Controls.Add(homeReuse);
            // Set the user control's Dock property to fill the parent panel
            homeReuse.Dock = DockStyle.Fill;

        }

        public void bmenu_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Menu";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = true;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = false;
            bclear.Visible = false;

            //new added for discount
            bupdatediscount.Visible = false;
            ldiscount.Visible = false;
            tbdiscount.Visible = false;
            lcurrentdiscount.Visible = false;
            //removing user control
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(resturantAvailability);
            //resize
            display.Size = new Size(736, 381);
            //for display text
            MenuLoad();

        }

        public void baddmenu_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Add/Delete Menu";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = true;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels and input menu panel
            ptb1.Visible = true;
            ptb2.Visible = true;
            ptb3.Visible = true;
            lfoodname.Visible = true;
            lfoodtype.Visible = true;
            lfoodprice.Visible = true;
            bupdate.Visible = true;
            tbfoodname.Visible = true;
            tbfoodtype.Visible = true;
            tbfoodprice.Visible = true;
            bclear.Visible = true;
            display.Visible = true;
            badd.Visible = true;
            bdelete.Visible = true;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;

            //new added for discount
            bupdatediscount.Visible = false;
            ldiscount.Visible = false;
            tbdiscount.Visible = false;
            lcurrentdiscount.Visible = false;
            //removing user control
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(resturantAvailability);
            //resize
            display.Size = new Size(736, 263);

            //for display text
            MenuLoad();


        }

        public void bprofits_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Profits";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = true;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = true;
            lincome.Visible = true;
            lprofit.Visible = true;
            display.Visible = true;
            badd.Visible = false;
            bdelete.Visible = false;
            bclear.Visible = false;

            //removing user control
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(resturantAvailability);
            //resize
            display.Size = new Size(736, 381);

            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT BILLDATE,ITEM_1,ITEM_2,ITEM_3,ITEM_4,ITEM_5,TOTAL,DISCOUNT FROM BILL WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            //now for total amount
            cmd = new SqlCommand("SELECT SUM(TOTAL) FROM BILL WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            double totalIncome = 0.0;
            try { totalIncome = double.Parse(cmd.ExecuteScalar().ToString()); } catch { totalIncome = 0.0; }
            con.Close();
            //database end here

            lincome.Text = "TOTAL REVENUE= " + Math.Round(totalIncome, 2).ToString();
            lcommison.Text = "HUNGRY HEROES COMMISON= " + Math.Round((totalIncome * 0.1), 2).ToString();
            lprofit.Text = "PROFIT= " + Math.Round((totalIncome - (totalIncome * 0.1)), 2).ToString();

        }

        public void blogout_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Logout";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = true;
            pb6.Visible = false;
            pb7.Visible = false;

            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;
            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;
            bclear.Visible = false;
            //removing user control
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(resturantAvailability);
            //new added for discount
            bupdatediscount.Visible = false;
            ldiscount.Visible = false;
            tbdiscount.Visible = false;
            lcurrentdiscount.Visible = false;

            //for confirming
            DialogResult result = MessageBox.Show("Are you sure to logout?", "Confirmation",
                               MessageBoxButtons.YesNo, MessageBoxIcon.Question);
            if (result == DialogResult.Yes)
            {
                this.Visible = false;
                ResturantLogin x = new ResturantLogin();
                x.Visible = true;
            }
            else
            {
                pb5.Visible = false;
                lTitles.Text = string.Empty;
            }
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void bdelete_Click(object sender, EventArgs e)
        {
            if (display.SelectedRows.Count > 0)
            {
                DialogResult result = MessageBox.Show("Confirm Deletion of the Selected Ttem?", "Confirmation",
                                 MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                if (result == DialogResult.Yes)
                {
                    //Database start
                    SqlConnection con = databaseConnection();
                    SqlCommand cmd = new SqlCommand("DELETE MENU WHERE FOOD_NAME = @foodname AND RESTURANT_NAME=@resturantname", con);
                    cmd.Parameters.AddWithValue("@foodname", tbfoodname.Text);
                    cmd.Parameters.AddWithValue("@resturantname", resturantname);
                    cmd.ExecuteNonQuery();
                    con.Close();
                    //Database closed

                    MessageBox.Show("Food Item Deleted Successfull", "Message",
                       MessageBoxButtons.OK, MessageBoxIcon.Information);

                    MenuLoad();

                    tbfoodname.Text = string.Empty;
                    tbfoodtype.Text = string.Empty;
                    tbfoodprice.Text = string.Empty;
                }

            }
            else
            {
                MessageBox.Show("Select an Item First to Delete!", "Alert",
                     MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }

        }

        public void badd_Click(object sender, EventArgs e)
        {

            if (tbfoodname.Text == "" || tbfoodprice.Text == "" || tbfoodtype.Text == "")
            {
                MessageBox.Show("Please Fill all the fields", "Alert",
                     MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
            }
            else
            {
                //Database start
                SqlConnection con = databaseConnection();
                SqlCommand cmd = new SqlCommand("SELECT * FROM MENU WHERE FOOD_NAME = @foodname AND RESTURANT_NAME=@resturantname", con);
                cmd.Parameters.AddWithValue("@foodname", tbfoodname.Text);
                cmd.Parameters.AddWithValue("@resturantname", resturantname);
                SqlDataAdapter da = new SqlDataAdapter(cmd);
                DataTable dt = new DataTable();
                da.Fill(dt);

                if (dt.Rows.Count > 0)
                {
                    MessageBox.Show("Food Item Already Exist!", "Message",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                else
                {
                    double foodPrice;
                    if (double.TryParse(tbfoodprice.Text, out foodPrice))
                    {
                        string query = "INSERT INTO MENU(FOOD_NAME, FOOD_PRICE, RESTURANT_NAME, FOOD_TYPE) VALUES (@name,@price,@resturantname,@type)";
                        cmd = new SqlCommand(query, con);
                        cmd.Parameters.AddWithValue("@name", tbfoodname.Text);
                        cmd.Parameters.AddWithValue("@price", foodPrice);
                        cmd.Parameters.AddWithValue("@resturantname", resturantname);
                        cmd.Parameters.AddWithValue("@type", tbfoodtype.Text);
                        cmd.ExecuteNonQuery();
                        con.Close();
                        //database end here
                        //refresh display
                        MenuLoad();

                        MessageBox.Show("Food Item Add Successfull", "Message",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);

                        tbfoodname.Text = string.Empty;
                        tbfoodtype.Text = string.Empty;
                        tbfoodprice.Text = string.Empty;
                    }
                    else
                    {
                        MessageBox.Show("Give numeric value to Price", "Message",
                            MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
            }
        }

        public void bupdate_Click(object sender, EventArgs e)
        {
            if (display.SelectedRows.Count > 0)
            {
                if (tbfoodname.Text != "" || tbfoodprice.Text != "" || tbfoodtype.Text != "")
                {
                    double foodPrice;
                    if (double.TryParse(tbfoodprice.Text, out foodPrice))
                    {
                        //Database Start
                        SqlConnection con = databaseConnection();
                        SqlCommand cmd = new SqlCommand("UPDATE MENU SET FOOD_NAME=@name, FOOD_PRICE=@price,FOOD_TYPE=@type WHERE FOOD_NAME = @foodname AND RESTURANT_NAME=@resturantname", con);
                        cmd.Parameters.AddWithValue("@foodname", foodname);
                        cmd.Parameters.AddWithValue("@resturantname", resturantname);
                        cmd.Parameters.AddWithValue("@name", tbfoodname.Text);
                        cmd.Parameters.AddWithValue("@price", foodPrice);
                        cmd.Parameters.AddWithValue("@type", tbfoodtype.Text);
                        cmd.ExecuteNonQuery();
                        con.Close();
                        //Database closed
                        MenuLoad();

                        MessageBox.Show("Food Item Update Successfull", "Message",
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                        tbfoodname.Text = string.Empty;
                        tbfoodtype.Text = string.Empty;
                        tbfoodprice.Text = string.Empty;
                    }
                    else
                    {
                        MessageBox.Show("Give numeric value to Price", "Message",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                }
                else
                {
                    MessageBox.Show("Please Fill All the fields!", "Warning!",
                                MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }

            }
            else
            {
                MessageBox.Show("Select an Item First to Update!", "Alert",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }


        }
        public void MenuLoad()
        {
            //Database start
            SqlConnection con = databaseConnection();
            SqlCommand cmd = new SqlCommand("SELECT FOOD_NAME,FOOD_PRICE,FOOD_TYPE FROM MENU WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            SqlDataAdapter da = new SqlDataAdapter(cmd);
            DataTable dt = new DataTable();
            da.Fill(dt);
            display.DataSource = dt;
            display.ClearSelection();
            con.Close();
            //database end here
        }
        public void ResturantHome_Load(object sender, EventArgs e)
        {
            int totalorder = 0;
            int totalrating = 0;
            //database start
            SqlConnection con = databaseConnection();

            //total order get
            SqlCommand cmd = new SqlCommand("SELECT TOTAL_ORDER FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            try { totalorder = int.Parse(cmd.ExecuteScalar().ToString()); } catch { }

            //total rating get
            cmd = new SqlCommand("SELECT TOTAL_RATING FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            try { totalrating = int.Parse(cmd.ExecuteScalar().ToString()); } catch { }
            con.Close();
            //database end here

            double rating = 0.0;
            try { rating = Math.Round((double)totalrating / totalorder, 2); } catch { }

            //rating set
            lratedisplay.Text = rating.ToString();

        }

        public void Cell_Click(object sender, DataGridViewCellEventArgs e)
        {
            if (e.RowIndex >= 0 && e.RowIndex < display.Rows.Count)
            {
                DataGridViewRow row = display.Rows[e.RowIndex];

                try
                {
                    tbfoodtype.Text = row.Cells["FOOD_TYPE"].Value.ToString().Trim();
                    tbfoodprice.Text = row.Cells["FOOD_PRICE"].Value.ToString().Trim();
                    tbfoodname.Text = row.Cells["FOOD_NAME"].Value.ToString().Trim();
                    //for updating, keeping this value
                    foodname = row.Cells["FOOD_NAME"].Value.ToString();
                }
                catch { }
            }
        }

        public void bclear_Click(object sender, EventArgs e)
        {
            tbfoodtype.Text = string.Empty;
            tbfoodprice.Text = string.Empty;
            tbfoodname.Text = string.Empty;
        }

        public void bstatus_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Status";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = true;
            pb7.Visible = false;

            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;
            bclear.Visible = false;

            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;

            //removing user control
            pHome.Controls.Remove(homeReuse);
            //Add user control to pHome Panel
            pHome.Controls.Add(resturantAvailability);
            // Set the user control's Dock property to fill the parent panel
            resturantAvailability.Dock = DockStyle.Fill;
            //new added for discount
            bupdatediscount.Visible = false;
            ldiscount.Visible = false;
            tbdiscount.Visible = false;
            lcurrentdiscount.Visible = false;
        }

        public void bdiscount_Click(object sender, EventArgs e)
        {
            //for Title text
            lTitles.Text = "Discount";
            //for button underline panel
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            pb7.Visible = true;

            //for buttons & lebels
            ptb1.Visible = false;
            ptb2.Visible = false;
            ptb3.Visible = false;
            lfoodname.Visible = false;
            lfoodtype.Visible = false;
            lfoodprice.Visible = false;
            bupdate.Visible = false;
            tbfoodname.Visible = false;
            tbfoodtype.Visible = false;
            tbfoodprice.Visible = false;
            lcommison.Visible = false;
            lincome.Visible = false;
            lprofit.Visible = false;
            bclear.Visible = false;

            display.Visible = false;
            badd.Visible = false;
            bdelete.Visible = false;

            //removing user control
            pHome.Controls.Remove(homeReuse);
            //Add user control to pHome Panel
            pHome.Controls.Remove(resturantAvailability);
            // Set the user control's Dock property to fill the parent panel
            resturantAvailability.Dock = DockStyle.Fill;

            //new added for discount
            bupdatediscount.Visible = true;
            ldiscount.Visible = true;
            tbdiscount.Visible = true;
            lcurrentdiscount.Visible = true;

            //database start
            SqlConnection con = databaseConnection();

            //get discount
            int discount = 0;
            SqlCommand cmd = new SqlCommand("SELECT DISCOUNT FROM RESTURANT_OWNER WHERE RESTURANT_NAME = @resturantname", con);
            cmd.Parameters.AddWithValue("@resturantname", resturantname);
            try { discount = int.Parse(cmd.ExecuteScalar().ToString()); } catch { discount = 0; }
            con.Close();
            //database end
            ldiscount.Text = discount + "%";
        }

        public void bupdatediscount_Click(object sender, EventArgs e)
        {
            if (tbdiscount.Text == "")
            {
                MessageBox.Show("Please Write the Percentage first to update!", "Warning!",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
            else if(int.TryParse(tbdiscount.Text,out int discountValue))
            {
                if(discountValue>=0 && discountValue <= 100)
                {
                    //database start
                    SqlConnection con = databaseConnection();

                    //set discount
                    SqlCommand cmd = new SqlCommand("UPDATE RESTURANT_OWNER SET DISCOUNT='"+ discountValue + "' WHERE RESTURANT_NAME = @resturantname", con);
                    cmd.Parameters.AddWithValue("@resturantname", resturantname);
                    cmd.ExecuteNonQuery();
                    con.Close();
                    //database end
                    tbdiscount.Text = string.Empty;
                    ldiscount.Text = discountValue + "%";
                }
                else
                {
                    MessageBox.Show("Enter a value between 0 to 100", "Warning!",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
            }
            else
            {
                MessageBox.Show("Enter Valid Number!", "Warning!",
                    MessageBoxButtons.OK, MessageBoxIcon.Warning);
            }
        }
    }
}
