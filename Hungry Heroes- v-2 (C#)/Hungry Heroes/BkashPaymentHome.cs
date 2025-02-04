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
using static System.Windows.Forms.VisualStyles.VisualStyleElement.ListView;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.StartPanel;

namespace Hungry_Heroes
{
    public partial class BkashPaymentHome : Form, IBkashPaymentHome
    {
        CustomerHomePage? customerHomePage;
        GuestHomePage? guestHomePage;
        CustomerOrder customerOrder;
        private double payment = 0;
        private string resturantName;
        private string name;
        private string address;
        public string username;
        private double ActualPayment=0.0;
        private double discount = 0;
        public BkashPaymentHome(CustomerOrder customerOrder, CustomerHomePage customerHomePage, double payment, string resturantName,double discount)
        {
            this.payment = payment;
            this.customerHomePage = customerHomePage;
            this.customerOrder = customerOrder;
            username = customerHomePage.username;
            InitializeComponent();
            this.discount=discount;
            bpay.Text = "Pay " + Math.Round(payment, 2) + " tk";
            this.resturantName = resturantName;

            //for checking if discounted or not
            for (int i = 0; i < customerOrder.price.Count; i++)
            {
                double itemTotal = customerOrder.price[i] * customerOrder.quantity[i];
                ActualPayment += itemTotal;               
            }

        }
        public BkashPaymentHome(CustomerOrder customerOrder, GuestHomePage guestHomePage, double payment, string resturantName, string name, string address)
        {
            this.payment = payment;
            this.guestHomePage = guestHomePage;
            this.customerOrder= customerOrder;
            InitializeComponent();
            bpay.Text = "Pay " + Math.Round(payment, 2) + " tk";
            this.resturantName = resturantName;
            //extra for guest
            this.name = name;
            this.address = address;
        }
        public SqlConnection databaseConnection()
        {
            //need to change connection string only here in all the other device which gonna use this project
            SqlConnection con = new SqlConnection(@"Data Source=(LocalDB)\MSSQLLocalDB;AttachDbFilename=""C:\Users\Faysal Ahmmed\Desktop\C# codes\Hungry Heroes\Hungry Heroes\MainDatabase.mdf"";Integrated Security=True");
            con.Open();
            return con;
        }
        public void bprevious_Click(object sender, EventArgs e)
        {
            if (customerHomePage != null)
            {
                CustomerOrder x = new CustomerOrder(customerHomePage, resturantName);
                this.Visible = false;
                x.Visible = true;
            }
            else
            {
                CustomerOrder x = new CustomerOrder(guestHomePage, resturantName);
                this.Visible = false;
                x.Visible = true;
            }
        }

        public void blogin_Click(object sender, EventArgs e)
        {
            string no = tbno.Text;
            string pin = tbpin.Text;

            if (no == "" || pin == "")
            {
                MessageBox.Show("Please Fillup all the Fields!", "Information",
                       MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            else
            {
                try
                {
                    long.Parse(no);
                    int.Parse(pin);

                    if (no.Length != 11)
                    {
                        MessageBox.Show("Account Number Should Contain 11 Digit!", "Information",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else if (pin.Length < 4)
                    {  
                        MessageBox.Show("Pin No Should have more than 4 Digit!", "Information",
                                MessageBoxButtons.OK, MessageBoxIcon.Information);
                    }
                    else
                    {
                        if (customerHomePage != null)
                        {
                            //database start
                            SqlConnection con = databaseConnection();

                            //exception checking
                            string item1 = customerOrder.foods[0].Trim() + "*" + customerOrder.quantity[0].ToString();
                            string? item2;
                            string? item3;
                            string? item4;
                            string? item5;

                            try { item2 = customerOrder.foods[1].Trim() + "*" + customerOrder.quantity[1].ToString(); }
                            catch (Exception no1) { item2 = null; }

                            try { item3 = customerOrder.foods[2].Trim() + "*" + customerOrder.quantity[2].ToString(); }
                            catch (Exception no1) { item3 = null; }

                            try { item4 = customerOrder.foods[3].Trim() + "*" + customerOrder.quantity[3].ToString(); }
                            catch (Exception no1) { item4 = null; }

                            try { item5 = customerOrder.foods[4].Trim() + "*" + customerOrder.quantity[4].ToString(); }
                            catch (Exception no1) { item5 = null; }

                            //GET USER ADDRESS
                            SqlCommand cmd = new SqlCommand("SELECT ADDRESS FROM CUSTOMER WHERE USERNAME = @username", con);
                            cmd.Parameters.AddWithValue("@username", username);
                            string customerAddress = cmd.ExecuteScalar().ToString();

                            //inserting value into bills table
                            string query = "INSERT INTO BILL(BILLDATE,ADDRESS,USERNAME,USER_TYPE,RESTURANT_NAME,ITEM_1,ITEM_2,ITEM_3,ITEM_4,ITEM_5,TOTAL,DISCOUNT)" +
                                "VALUES (@date,@address,@username, @type,@resturantname,@item1,@item2,@item3,@item4,@item5,@total,@discount)";
                            cmd = new SqlCommand(query, con);
                            cmd.Parameters.AddWithValue("@address", customerAddress);
                            cmd.Parameters.AddWithValue("@username", username);
                            cmd.Parameters.AddWithValue("@type", "CUSTOMER");
                            cmd.Parameters.AddWithValue("@resturantname", resturantName);
                            cmd.Parameters.AddWithValue("@item1", item1);
                            //checking null or not. if null inserting null vale
                            if (item2 != null)
                            {
                                cmd.Parameters.AddWithValue("@item2", item2);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item2", DBNull.Value);
                            }
                            //
                            if (item3 != null)
                            {
                                cmd.Parameters.AddWithValue("@item3", item3);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item3", DBNull.Value);
                            }
                            //
                            if (item4 != null)
                            {
                                cmd.Parameters.AddWithValue("@item4", item4);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item4", DBNull.Value);
                            }
                            //
                            if (item5 != null)
                            {
                                cmd.Parameters.AddWithValue("@item5", item5);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item5", DBNull.Value);
                            }
                            cmd.Parameters.AddWithValue("@total", Math.Round(payment, 2));
                            cmd.Parameters.AddWithValue("@date", DateTime.Now);
                            //setting discount value
                            if (ActualPayment>payment)
                            {
                                discount = 1 - discount;
                                discount = discount * 100;
                                cmd.Parameters.AddWithValue("@discount", Math.Round(discount).ToString()+"%");
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@discount", DBNull.Value);
                            }

                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database close

                            PaymentSuccess x = new PaymentSuccess(customerHomePage, Math.Round(payment, 2), resturantName);
                            this.Visible = false;
                            x.Visible = true;
                        }
                        else
                        {
                            //database start
                            SqlConnection con = databaseConnection();
                            string query = "INSERT INTO GUEST(GUEST_ADDRESS,GUEST_NAME) VALUES (@address, @name)";
                            SqlCommand cmd = new SqlCommand(query, con);
                            cmd.Parameters.AddWithValue("@address", address);
                            cmd.Parameters.AddWithValue("@name", name);
                            cmd.ExecuteNonQuery();

                            //now for login details table
                            query = "INSERT INTO LOGIN_DETAILS(TIME,TYPE) VALUES (@time, @type)";
                            cmd = new SqlCommand(query, con);
                            cmd.Parameters.AddWithValue("@time", DateTime.Now);
                            cmd.Parameters.AddWithValue("@type", "GUEST");
                            cmd.ExecuteNonQuery();
                            //exception checking
                            string item1 = customerOrder.foods[0].Trim() + "*" + customerOrder.quantity[0].ToString();
                            string? item2;
                            string? item3;
                            string? item4;
                            string? item5;

                            try { item2 = customerOrder.foods[1].Trim() + "*" + customerOrder.quantity[1].ToString(); }
                            catch (Exception no1) { item2 = null; }

                            try { item3 = customerOrder.foods[2].Trim() + "*" + customerOrder.quantity[2].ToString(); }
                            catch (Exception no1) { item3 = null; }

                            try { item4 = customerOrder.foods[3].Trim() + "*" + customerOrder.quantity[3].ToString(); }
                            catch (Exception no1) { item4 = null; }

                            try { item5 = customerOrder.foods[4].Trim() + "*" + customerOrder.quantity[4].ToString(); }
                            catch (Exception no1) { item5 = null; }

                            //inserting value into bills table
                            query = "INSERT INTO BILL(BILLDATE,ADDRESS,USER_TYPE,RESTURANT_NAME,ITEM_1,ITEM_2,ITEM_3,ITEM_4,ITEM_5,TOTAL)" +
                                "VALUES (@date,@address,@type,@resturantname,@item1,@item2,@item3,@item4,@item5,@total)";
                            cmd = new SqlCommand(query, con);
                            cmd.Parameters.AddWithValue("@address", address);
                            cmd.Parameters.AddWithValue("@type", "GUEST");
                            cmd.Parameters.AddWithValue("@resturantname", resturantName);
                            cmd.Parameters.AddWithValue("@item1", item1);
                            //checking null or not. if null inserting null vale
                            if (item2 != null)
                            {
                                cmd.Parameters.AddWithValue("@item2", item2);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item2", DBNull.Value);
                            }
                            //
                            if (item3 != null)
                            {
                                cmd.Parameters.AddWithValue("@item3", item3);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item3", DBNull.Value);
                            }
                            //
                            if (item4 != null)
                            {
                                cmd.Parameters.AddWithValue("@item4", item4);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item4", DBNull.Value);
                            }
                            //
                            if (item5 != null)
                            {
                                cmd.Parameters.AddWithValue("@item5", item5);
                            }
                            else
                            {
                                cmd.Parameters.AddWithValue("@item5", DBNull.Value);
                            }
                            cmd.Parameters.AddWithValue("@total", Math.Round(payment, 2));
                            cmd.Parameters.AddWithValue("@date", DateTime.Now);

                            cmd.ExecuteNonQuery();
                            con.Close();
                            //database close

                            PaymentSuccess x = new PaymentSuccess(guestHomePage, Math.Round(payment, 2), resturantName);
                            this.Visible = false;
                            x.Visible = true;
                        }
                    }

                }
                catch (Exception f)
                {
                    MessageBox.Show("Please Give Correct Information!+"+f.Message, "Information",
                                      MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
        }
    }
}


