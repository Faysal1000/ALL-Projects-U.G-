using Hungry_Heroes.Interfaces;

namespace Hungry_Heroes
{
    public partial class First_page : Form, IFirstPage
    {
        public First_page()
        {
            InitializeComponent();
        }

        public void bcustomerlogin_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            CustomerLogin x = new CustomerLogin();
            x.Visible = true;
        }

        public void bresturantlogin_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            ResturantLogin x = new ResturantLogin();
            x.Visible = true;
        }

        public void bcustomersignup_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            CustomerSignup x = new CustomerSignup();
            x.Visible = true;
        }

        public void bresturantsignup_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            ResturantSignup x = new ResturantSignup();
            x.Visible = true;
        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void badmin_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            AdminLogin x = new AdminLogin();
            x.Visible = true;
        }
        public void bmanager_Click(object sender, EventArgs e)
        {
            this.Visible = false;
            ManagerLogin x = new ManagerLogin();
            x.Visible = true;

        }
        public void lguest_LinkClicked(object sender, LinkLabelLinkClickedEventArgs e)
        {
            this.Visible = false;
            GuestHomePage x = new GuestHomePage();
            x.Visible = true;
        }
    }
}