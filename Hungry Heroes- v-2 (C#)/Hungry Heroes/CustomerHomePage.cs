using Hungry_Heroes.Interfaces;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Hungry_Heroes
{
    public partial class CustomerHomePage : Form, ICustomerHomePage
    {
        private int colorFlag = 0;
        //color code
        private string[] colorArr = { "#1C4E80", "#0091D5", "#B3C100", "#1F3F49", "#Ac3E31","#361027",
                              "#56a1bf","#3a5c85","#1b1853","#19102e","#b42313","#222d81","#74341a","#97dbd2" };
        //for generating random color method
        public Color getColor()
        {
            Random random = new Random();
            int index = random.Next(0, colorArr.Length);
            Color color = ColorTranslator.FromHtml(colorArr[index]);
            return color;
        }
        //declaring for reusable panel
        private HomeReuse homeReuse;
        private FindResturantReuse findResturantReuse;
        private PreviousOrderReuse previousOrderReuse;
        private ProfileReuse profileReuse;
        private AboutUsReuse aboutUsReuse;
        public string username;
        public CustomerHomePage(string username)
        {
            InitializeComponent();
            this.username = username;
            //instanciating usercontrol
            homeReuse = new HomeReuse();
            findResturantReuse = new FindResturantReuse(this);
            previousOrderReuse = new PreviousOrderReuse(username);
            profileReuse = new ProfileReuse(this);
            aboutUsReuse = new AboutUsReuse();
            pHome.Controls.Add(homeReuse);
            pb1.Visible = true;
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
            //for theme visible false
            ltheme.Visible = false;
            darkmode.Visible = false;
            whitemode.Visible = false;
            randomcolor.Visible = false;
            bapplytheme.Visible = false;

            //Add user control to pHome Panel
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(aboutUsReuse);
            pHome.Controls.Remove(profileReuse);
            pHome.Controls.Remove(findResturantReuse);
            pHome.Controls.Remove(previousOrderReuse);
            pHome.Controls.Add(homeReuse);
            // Set the user control's Dock property to fill the parent panel
            homeReuse.Dock = DockStyle.Fill;
        }

        public void bfres_Click(object sender, EventArgs e)
        {
            lTitles.Text = "Find Resturant";
            pb1.Visible = false;
            pb2.Visible = true;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;

            //for theme visible false
            ltheme.Visible = false;
            darkmode.Visible = false;
            whitemode.Visible = false;
            randomcolor.Visible = false;
            bapplytheme.Visible = false;
            //
            pHome.Controls.Remove(aboutUsReuse);
            pHome.Controls.Remove(profileReuse);
            pHome.Controls.Remove(previousOrderReuse);
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Add(findResturantReuse);
            // Set the user control's Dock property to fill the parent panel
            findResturantReuse.Dock = DockStyle.Fill;
        }

        public void bprofile_Click(object sender, EventArgs e)
        {
            lTitles.Text = "Profile";
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = true;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = false;
            //for theme visible false
            ltheme.Visible = false;
            darkmode.Visible = false;
            whitemode.Visible = false;
            randomcolor.Visible = false;
            bapplytheme.Visible = false;
            //
            pHome.Controls.Remove(aboutUsReuse);
            pHome.Controls.Remove(previousOrderReuse);
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(findResturantReuse);
            //refresh
            profileReuse = new ProfileReuse(this);
            pHome.Controls.Add(profileReuse);
            // Set the user control's Dock property to fill the parent panel
            profileReuse.Dock = DockStyle.Fill;

        }

        public void bporder_Click(object sender, EventArgs e)
        {
            lTitles.Text = "Previous Order";
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = true;
            pb5.Visible = false;
            pb6.Visible = false;
            //for theme visible false
            ltheme.Visible = false;
            darkmode.Visible = false;
            whitemode.Visible = false;
            randomcolor.Visible = false;
            bapplytheme.Visible = false;
            //
            pHome.Controls.Remove(aboutUsReuse);
            pHome.Controls.Remove(profileReuse);
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(findResturantReuse);
            //refresh
            previousOrderReuse = new PreviousOrderReuse(username);
            pHome.Controls.Add(previousOrderReuse);
            // Set the user control's Dock property to fill the parent panel
            previousOrderReuse.Dock = DockStyle.Fill;

        }

        public void bthemes_Click(object sender, EventArgs e)
        {
            lTitles.Text = "Themes";
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = true;
            pb6.Visible = false;
            //
            ltheme.Visible = true;
            darkmode.Visible = true;
            whitemode.Visible = true;
            randomcolor.Visible = true;
            bapplytheme.Visible = true;
            //
            pHome.Controls.Remove(aboutUsReuse);
            pHome.Controls.Remove(profileReuse);
            pHome.Controls.Remove(previousOrderReuse);
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(findResturantReuse);

        }

        public void baboutus_Click(object sender, EventArgs e)
        {
            lTitles.Text = "About Us";
            pb1.Visible = false;
            pb2.Visible = false;
            pb3.Visible = false;
            pb4.Visible = false;
            pb5.Visible = false;
            pb6.Visible = true;
            //for theme visible false
            ltheme.Visible = false;
            darkmode.Visible = false;
            whitemode.Visible = false;
            randomcolor.Visible = false;
            bapplytheme.Visible = false;
            //
            pHome.Controls.Remove(profileReuse);
            pHome.Controls.Remove(previousOrderReuse);
            pHome.Controls.Remove(homeReuse);
            pHome.Controls.Remove(findResturantReuse);
            pHome.Controls.Add(aboutUsReuse);
            // Set the user control's Dock property to fill the parent panel
            aboutUsReuse.Dock = DockStyle.Fill;

        }

        public void bexit_Click(object sender, EventArgs e)
        {
            Application.Exit();
        }

        public void bapplytheme_Click(object sender, EventArgs e)
        {
            if (darkmode.Checked)
            {
                colorFlag = 0;
            }
            else if (whitemode.Checked)
            {
                colorFlag = 1;
            }
            else if (randomcolor.Checked)
            {
                colorFlag = 2;
            }
            //
            Color phomecolor;
            if (pHome.BackColor != Color.White)
            {
                phomecolor = pHome.BackColor;
            }
            else { phomecolor = Color.Gray; }
            //
            if (colorFlag == 0)
            {
                bhome.BackColor = Color.Black;
                bfres.BackColor = Color.Black;
                bprofile.BackColor = Color.Black;
                bporder.BackColor = Color.Black;
                bthemes.BackColor = Color.Black;
                baboutus.BackColor = Color.Black;
                bexit.BackColor = Color.Black;
                buttonpanel.BackColor = Color.Black;
                ptop.BackColor = Color.Black;

                lTitles.ForeColor = Color.White;
                lhh.ForeColor = Color.White;
                pHome.BackColor = phomecolor;

                bhome.ForeColor = Color.White;
                bfres.ForeColor = Color.White;
                bprofile.ForeColor = Color.White;
                bporder.ForeColor = Color.White;
                bthemes.ForeColor = Color.White;
                baboutus.ForeColor = Color.White;
                bexit.ForeColor = Color.White;
            }
            else if (colorFlag == 1)
            {
                bhome.BackColor = Color.White;
                bfres.BackColor = Color.White;
                bprofile.BackColor = Color.White;
                bporder.BackColor = Color.White;
                bthemes.BackColor = Color.White;
                baboutus.BackColor = Color.White;
                bexit.BackColor = Color.White;
                buttonpanel.BackColor = Color.White;
                ptop.BackColor = Color.White;

                lTitles.ForeColor = Color.Black;
                lhh.ForeColor = Color.Black;
                pHome.BackColor = Color.White;

                bhome.ForeColor = Color.Black;
                bfres.ForeColor = Color.Black;
                bprofile.ForeColor = Color.Black;
                bporder.ForeColor = Color.Black;
                bthemes.ForeColor = Color.Black;
                baboutus.ForeColor = Color.Black;
                bexit.ForeColor = Color.Black;

            }
            else if (colorFlag == 2)
            {
                Color c = getColor();
                bhome.BackColor = c;
                bfres.BackColor = c;
                bprofile.BackColor = c;
                bporder.BackColor = c;
                bthemes.BackColor = c;
                baboutus.BackColor = c;
                bexit.BackColor = c;
                buttonpanel.BackColor = c;
                ptop.BackColor = c;

                lTitles.ForeColor = Color.White;
                lhh.ForeColor = Color.White;
                pHome.BackColor = phomecolor;

                bhome.ForeColor = Color.White;
                bfres.ForeColor = Color.White;
                bprofile.ForeColor = Color.White;
                bporder.ForeColor = Color.White;
                bthemes.ForeColor = Color.White;
                baboutus.ForeColor = Color.White;
                bexit.ForeColor = Color.White;
            }
        }
    }
}
