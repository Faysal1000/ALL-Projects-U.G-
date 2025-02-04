namespace Hungry_Heroes
{
    partial class ResturantSignup
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ResturantSignup));
            panel1 = new Panel();
            tbphone = new TextBox();
            lcontact = new Label();
            tbresname = new TextBox();
            lresturantname = new Label();
            tbcpass = new TextBox();
            bexit = new Button();
            bprevious = new Button();
            bsignup = new Button();
            lconfirmpass = new Label();
            lcaptcha = new Label();
            tbcans = new TextBox();
            lcpequal = new Label();
            lcp2 = new Label();
            lcpplus = new Label();
            lcp1 = new Label();
            tbemail = new TextBox();
            label2 = new Label();
            tbname = new TextBox();
            lname = new Label();
            tbpass = new TextBox();
            tbusername = new TextBox();
            lpass = new Label();
            lusername = new Label();
            brecapta = new Button();
            panel1.SuspendLayout();
            SuspendLayout();
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.None;
            panel1.BackColor = Color.Transparent;
            panel1.BorderStyle = BorderStyle.Fixed3D;
            panel1.Controls.Add(brecapta);
            panel1.Controls.Add(tbphone);
            panel1.Controls.Add(lcontact);
            panel1.Controls.Add(tbresname);
            panel1.Controls.Add(lresturantname);
            panel1.Controls.Add(tbcpass);
            panel1.Controls.Add(bexit);
            panel1.Controls.Add(bprevious);
            panel1.Controls.Add(bsignup);
            panel1.Controls.Add(lconfirmpass);
            panel1.Controls.Add(lcaptcha);
            panel1.Controls.Add(tbcans);
            panel1.Controls.Add(lcpequal);
            panel1.Controls.Add(lcp2);
            panel1.Controls.Add(lcpplus);
            panel1.Controls.Add(lcp1);
            panel1.Controls.Add(tbemail);
            panel1.Controls.Add(label2);
            panel1.Controls.Add(tbname);
            panel1.Controls.Add(lname);
            panel1.Controls.Add(tbpass);
            panel1.Controls.Add(tbusername);
            panel1.Controls.Add(lpass);
            panel1.Controls.Add(lusername);
            panel1.Location = new Point(237, 25);
            panel1.Margin = new Padding(4);
            panel1.Name = "panel1";
            panel1.Size = new Size(409, 485);
            panel1.TabIndex = 0;
            // 
            // tbphone
            // 
            tbphone.Anchor = AnchorStyles.None;
            tbphone.BackColor = SystemColors.ControlLight;
            tbphone.Cursor = Cursors.IBeam;
            tbphone.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbphone.Location = new Point(166, 168);
            tbphone.MaxLength = 11;
            tbphone.Name = "tbphone";
            tbphone.PlaceholderText = "Your Phone Number";
            tbphone.ScrollBars = ScrollBars.Horizontal;
            tbphone.Size = new Size(199, 23);
            tbphone.TabIndex = 67;
            // 
            // lcontact
            // 
            lcontact.Anchor = AnchorStyles.None;
            lcontact.AutoSize = true;
            lcontact.BackColor = Color.Transparent;
            lcontact.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lcontact.ForeColor = Color.White;
            lcontact.Image = (Image)resources.GetObject("lcontact.Image");
            lcontact.ImageAlign = ContentAlignment.MiddleLeft;
            lcontact.Location = new Point(36, 170);
            lcontact.Name = "lcontact";
            lcontact.Size = new Size(121, 20);
            lcontact.TabIndex = 66;
            lcontact.Text = "      Contact No";
            lcontact.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbresname
            // 
            tbresname.Anchor = AnchorStyles.None;
            tbresname.BackColor = SystemColors.ControlLight;
            tbresname.Cursor = Cursors.IBeam;
            tbresname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbresname.Location = new Point(166, 202);
            tbresname.MaxLength = 25;
            tbresname.Name = "tbresname";
            tbresname.PlaceholderText = "Your Resturant Name";
            tbresname.ScrollBars = ScrollBars.Horizontal;
            tbresname.Size = new Size(199, 23);
            tbresname.TabIndex = 65;
            // 
            // lresturantname
            // 
            lresturantname.Anchor = AnchorStyles.None;
            lresturantname.AutoSize = true;
            lresturantname.BackColor = Color.Transparent;
            lresturantname.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lresturantname.ForeColor = Color.LightYellow;
            lresturantname.Image = (Image)resources.GetObject("lresturantname.Image");
            lresturantname.ImageAlign = ContentAlignment.MiddleLeft;
            lresturantname.Location = new Point(3, 203);
            lresturantname.Name = "lresturantname";
            lresturantname.Size = new Size(153, 20);
            lresturantname.TabIndex = 64;
            lresturantname.Text = "     Resturnat Name";
            lresturantname.TextAlign = ContentAlignment.MiddleLeft;
            // 
            // tbcpass
            // 
            tbcpass.Anchor = AnchorStyles.None;
            tbcpass.BackColor = SystemColors.ControlLight;
            tbcpass.Cursor = Cursors.IBeam;
            tbcpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbcpass.Location = new Point(167, 303);
            tbcpass.MaxLength = 25;
            tbcpass.Name = "tbcpass";
            tbcpass.PlaceholderText = "Retype Your Password";
            tbcpass.ScrollBars = ScrollBars.Horizontal;
            tbcpass.Size = new Size(199, 23);
            tbcpass.TabIndex = 63;
            tbcpass.UseSystemPasswordChar = true;
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.BackColor = Color.DarkGray;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bexit.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bexit.FlatAppearance.MouseOverBackColor = Color.Red;
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(87, 405);
            bexit.Name = "bexit";
            bexit.Size = new Size(245, 34);
            bexit.TabIndex = 62;
            bexit.Text = "   Exit";
            bexit.UseVisualStyleBackColor = false;
            bexit.Click += bexit_Click;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.BackColor = Color.DarkGray;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bprevious.FlatAppearance.MouseDownBackColor = Color.DarkRed;
            bprevious.FlatAppearance.MouseOverBackColor = Color.Tomato;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(214, 364);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(119, 34);
            bprevious.TabIndex = 61;
            bprevious.Text = "   Previous";
            bprevious.UseVisualStyleBackColor = false;
            bprevious.Click += bprevious_Click;
            // 
            // bsignup
            // 
            bsignup.Anchor = AnchorStyles.None;
            bsignup.BackColor = Color.DarkGray;
            bsignup.BackgroundImageLayout = ImageLayout.Stretch;
            bsignup.Cursor = Cursors.Hand;
            bsignup.FlatAppearance.BorderColor = Color.FromArgb(192, 0, 0);
            bsignup.FlatAppearance.MouseDownBackColor = Color.LightGreen;
            bsignup.FlatAppearance.MouseOverBackColor = Color.Lime;
            bsignup.FlatStyle = FlatStyle.Flat;
            bsignup.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            bsignup.Image = Properties.Resources.login;
            bsignup.ImageAlign = ContentAlignment.MiddleLeft;
            bsignup.Location = new Point(87, 364);
            bsignup.Name = "bsignup";
            bsignup.Size = new Size(119, 34);
            bsignup.TabIndex = 60;
            bsignup.Text = "   Signup";
            bsignup.UseVisualStyleBackColor = false;
            bsignup.Click += bsignup_Click;
            // 
            // lconfirmpass
            // 
            lconfirmpass.Anchor = AnchorStyles.None;
            lconfirmpass.AutoSize = true;
            lconfirmpass.BackColor = Color.Transparent;
            lconfirmpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lconfirmpass.ForeColor = Color.White;
            lconfirmpass.Image = (Image)resources.GetObject("lconfirmpass.Image");
            lconfirmpass.ImageAlign = ContentAlignment.MiddleLeft;
            lconfirmpass.Location = new Point(3, 303);
            lconfirmpass.Name = "lconfirmpass";
            lconfirmpass.Size = new Size(160, 20);
            lconfirmpass.TabIndex = 59;
            lconfirmpass.Text = "    Confirm Password";
            lconfirmpass.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lcaptcha
            // 
            lcaptcha.Anchor = AnchorStyles.None;
            lcaptcha.AutoSize = true;
            lcaptcha.BackColor = Color.Transparent;
            lcaptcha.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lcaptcha.ForeColor = SystemColors.Control;
            lcaptcha.Image = (Image)resources.GetObject("lcaptcha.Image");
            lcaptcha.ImageAlign = ContentAlignment.MiddleLeft;
            lcaptcha.Location = new Point(61, 239);
            lcaptcha.Name = "lcaptcha";
            lcaptcha.Size = new Size(94, 20);
            lcaptcha.TabIndex = 58;
            lcaptcha.Text = "     Captcha";
            lcaptcha.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbcans
            // 
            tbcans.Anchor = AnchorStyles.None;
            tbcans.BackColor = SystemColors.ControlLight;
            tbcans.Cursor = Cursors.IBeam;
            tbcans.Font = new Font("Calibri", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            tbcans.Location = new Point(301, 233);
            tbcans.MaxLength = 3;
            tbcans.Name = "tbcans";
            tbcans.PlaceholderText = "?";
            tbcans.ScrollBars = ScrollBars.Horizontal;
            tbcans.Size = new Size(39, 26);
            tbcans.TabIndex = 57;
            // 
            // lcpequal
            // 
            lcpequal.Anchor = AnchorStyles.None;
            lcpequal.AutoSize = true;
            lcpequal.BackColor = Color.Black;
            lcpequal.BorderStyle = BorderStyle.Fixed3D;
            lcpequal.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpequal.ForeColor = Color.WhiteSmoke;
            lcpequal.ImageAlign = ContentAlignment.MiddleLeft;
            lcpequal.Location = new Point(272, 235);
            lcpequal.Name = "lcpequal";
            lcpequal.Size = new Size(23, 23);
            lcpequal.TabIndex = 56;
            lcpequal.Text = "=";
            lcpequal.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp2
            // 
            lcp2.Anchor = AnchorStyles.None;
            lcp2.AutoSize = true;
            lcp2.BackColor = Color.Black;
            lcp2.BorderStyle = BorderStyle.Fixed3D;
            lcp2.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp2.ForeColor = Color.WhiteSmoke;
            lcp2.ImageAlign = ContentAlignment.MiddleLeft;
            lcp2.Location = new Point(232, 235);
            lcp2.Name = "lcp2";
            lcp2.Size = new Size(2, 23);
            lcp2.TabIndex = 55;
            lcp2.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcpplus
            // 
            lcpplus.Anchor = AnchorStyles.None;
            lcpplus.AutoSize = true;
            lcpplus.BackColor = Color.Black;
            lcpplus.BorderStyle = BorderStyle.Fixed3D;
            lcpplus.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcpplus.ForeColor = Color.WhiteSmoke;
            lcpplus.ImageAlign = ContentAlignment.MiddleLeft;
            lcpplus.Location = new Point(204, 235);
            lcpplus.Name = "lcpplus";
            lcpplus.Size = new Size(23, 23);
            lcpplus.TabIndex = 54;
            lcpplus.Text = "+";
            lcpplus.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // lcp1
            // 
            lcp1.Anchor = AnchorStyles.None;
            lcp1.AutoSize = true;
            lcp1.BackColor = Color.Black;
            lcp1.BorderStyle = BorderStyle.Fixed3D;
            lcp1.Font = new Font("Algerian", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcp1.ForeColor = Color.WhiteSmoke;
            lcp1.ImageAlign = ContentAlignment.MiddleLeft;
            lcp1.Location = new Point(166, 235);
            lcp1.Name = "lcp1";
            lcp1.Size = new Size(2, 23);
            lcp1.TabIndex = 53;
            lcp1.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // tbemail
            // 
            tbemail.Anchor = AnchorStyles.None;
            tbemail.BackColor = SystemColors.ControlLight;
            tbemail.Cursor = Cursors.IBeam;
            tbemail.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbemail.Location = new Point(166, 130);
            tbemail.MaxLength = 40;
            tbemail.Name = "tbemail";
            tbemail.PlaceholderText = "Your Email Adress";
            tbemail.ScrollBars = ScrollBars.Horizontal;
            tbemail.Size = new Size(199, 23);
            tbemail.TabIndex = 52;
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            label2.ForeColor = Color.White;
            label2.Image = (Image)resources.GetObject("label2.Image");
            label2.ImageAlign = ContentAlignment.MiddleLeft;
            label2.Location = new Point(78, 130);
            label2.Name = "label2";
            label2.Size = new Size(78, 20);
            label2.TabIndex = 51;
            label2.Text = "      Email";
            label2.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.ControlLight;
            tbname.Cursor = Cursors.IBeam;
            tbname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbname.Location = new Point(166, 50);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Your Full Name";
            tbname.ScrollBars = ScrollBars.Horizontal;
            tbname.Size = new Size(199, 23);
            tbname.TabIndex = 47;
            // 
            // lname
            // 
            lname.Anchor = AnchorStyles.None;
            lname.AutoSize = true;
            lname.BackColor = Color.Transparent;
            lname.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lname.ForeColor = Color.White;
            lname.Image = (Image)resources.GetObject("lname.Image");
            lname.ImageAlign = ContentAlignment.MiddleLeft;
            lname.Location = new Point(75, 50);
            lname.Name = "lname";
            lname.Size = new Size(81, 20);
            lname.TabIndex = 49;
            lname.Text = "      Name";
            lname.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ControlLight;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(167, 270);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Your Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(199, 23);
            tbpass.TabIndex = 48;
            tbpass.UseSystemPasswordChar = true;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ControlLight;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(166, 90);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Your Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 50;
            // 
            // lpass
            // 
            lpass.Anchor = AnchorStyles.None;
            lpass.AutoSize = true;
            lpass.BackColor = Color.Transparent;
            lpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lpass.ForeColor = Color.White;
            lpass.Image = (Image)resources.GetObject("lpass.Image");
            lpass.ImageAlign = ContentAlignment.MiddleLeft;
            lpass.Location = new Point(57, 271);
            lpass.Name = "lpass";
            lpass.Size = new Size(103, 20);
            lpass.TabIndex = 46;
            lpass.Text = "     Password";
            lpass.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Bold, GraphicsUnit.Point);
            lusername.ForeColor = Color.White;
            lusername.Image = Properties.Resources.username1;
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(48, 90);
            lusername.Name = "lusername";
            lusername.Size = new Size(112, 20);
            lusername.TabIndex = 45;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // brecapta
            // 
            brecapta.Anchor = AnchorStyles.None;
            brecapta.BackColor = SystemColors.ScrollBar;
            brecapta.BackgroundImageLayout = ImageLayout.Stretch;
            brecapta.Cursor = Cursors.Hand;
            brecapta.FlatStyle = FlatStyle.Popup;
            brecapta.Image = (Image)resources.GetObject("brecapta.Image");
            brecapta.Location = new Point(340, 233);
            brecapta.Name = "brecapta";
            brecapta.Size = new Size(25, 26);
            brecapta.TabIndex = 82;
            brecapta.UseMnemonic = false;
            brecapta.UseVisualStyleBackColor = false;
            brecapta.Click += brecapta_Click;
            // 
            // ResturantSignup
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(884, 561);
            Controls.Add(panel1);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "ResturantSignup";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "ResturantSignup";
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            ResumeLayout(false);
        }

        #endregion

        private Panel panel1;
        private TextBox tbcpass;
        private Button bexit;
        private Button bprevious;
        private Button bsignup;
        private Label lconfirmpass;
        private Label lcaptcha;
        private TextBox tbcans;
        private Label lcpequal;
        private Label lcp2;
        private Label lcpplus;
        private Label lcp1;
        private TextBox tbemail;
        private Label label2;
        private TextBox tbname;
        private Label lname;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label lpass;
        private Label lusername;
        private TextBox tbresname;
        private Label lresturantname;
        private TextBox tbphone;
        private Label lcontact;
        private Button brecapta;
    }
}