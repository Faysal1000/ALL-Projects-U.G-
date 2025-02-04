namespace Hungry_Heroes
{
    partial class CustomerSignup
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CustomerSignup));
            tbpass = new TextBox();
            tbusername = new TextBox();
            lpass = new Label();
            lusername = new Label();
            tbname = new TextBox();
            lname = new Label();
            birthdaypicker = new DateTimePicker();
            lbdate = new Label();
            tbemail = new TextBox();
            label2 = new Label();
            cbsq = new ComboBox();
            lsq = new Label();
            lsqa = new Label();
            tbsqa = new TextBox();
            lcp1 = new Label();
            lcpplus = new Label();
            lcp2 = new Label();
            lcpequal = new Label();
            tbcans = new TextBox();
            lcaptcha = new Label();
            lconfirmpass = new Label();
            bexit = new Button();
            bprevious = new Button();
            bsignup = new Button();
            tbcpass = new TextBox();
            tbaddress = new TextBox();
            label1 = new Label();
            brecapta = new Button();
            SuspendLayout();
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ControlLight;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(258, 370);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Your Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(199, 23);
            tbpass.TabIndex = 18;
            tbpass.UseSystemPasswordChar = true;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ControlLight;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(257, 95);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Your Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 22;
            // 
            // lpass
            // 
            lpass.Anchor = AnchorStyles.None;
            lpass.AutoSize = true;
            lpass.BackColor = Color.Transparent;
            lpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lpass.ForeColor = Color.Black;
            lpass.Image = (Image)resources.GetObject("lpass.Image");
            lpass.ImageAlign = ContentAlignment.MiddleLeft;
            lpass.Location = new Point(162, 371);
            lpass.Name = "lpass";
            lpass.Size = new Size(90, 20);
            lpass.TabIndex = 16;
            lpass.Text = "     Password";
            lpass.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lusername.ForeColor = Color.Black;
            lusername.Image = Properties.Resources.username1;
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(153, 95);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 15;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.ControlLight;
            tbname.Cursor = Cursors.IBeam;
            tbname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbname.Location = new Point(257, 55);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Your Full Name";
            tbname.ScrollBars = ScrollBars.Horizontal;
            tbname.Size = new Size(199, 23);
            tbname.TabIndex = 17;
            // 
            // lname
            // 
            lname.Anchor = AnchorStyles.None;
            lname.AutoSize = true;
            lname.BackColor = Color.Transparent;
            lname.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lname.ForeColor = Color.Black;
            lname.Image = (Image)resources.GetObject("lname.Image");
            lname.ImageAlign = ContentAlignment.MiddleLeft;
            lname.Location = new Point(180, 55);
            lname.Name = "lname";
            lname.Size = new Size(71, 20);
            lname.TabIndex = 21;
            lname.Text = "      Name";
            lname.TextAlign = ContentAlignment.MiddleRight;
            // 
            // birthdaypicker
            // 
            birthdaypicker.Anchor = AnchorStyles.None;
            birthdaypicker.Cursor = Cursors.Hand;
            birthdaypicker.Location = new Point(256, 173);
            birthdaypicker.MaxDate = new DateTime(2029, 7, 5, 0, 0, 0, 0);
            birthdaypicker.Name = "birthdaypicker";
            birthdaypicker.Size = new Size(200, 23);
            birthdaypicker.TabIndex = 23;
            // 
            // lbdate
            // 
            lbdate.Anchor = AnchorStyles.None;
            lbdate.AutoSize = true;
            lbdate.BackColor = Color.Transparent;
            lbdate.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lbdate.ForeColor = Color.Black;
            lbdate.Image = (Image)resources.GetObject("lbdate.Image");
            lbdate.ImageAlign = ContentAlignment.MiddleLeft;
            lbdate.Location = new Point(152, 173);
            lbdate.Name = "lbdate";
            lbdate.Size = new Size(98, 20);
            lbdate.TabIndex = 24;
            lbdate.Text = "      Birth-Date";
            lbdate.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbemail
            // 
            tbemail.Anchor = AnchorStyles.None;
            tbemail.BackColor = SystemColors.ControlLight;
            tbemail.Cursor = Cursors.IBeam;
            tbemail.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbemail.Location = new Point(257, 135);
            tbemail.MaxLength = 40;
            tbemail.Name = "tbemail";
            tbemail.PlaceholderText = "Your Email Adress";
            tbemail.ScrollBars = ScrollBars.Horizontal;
            tbemail.Size = new Size(199, 23);
            tbemail.TabIndex = 26;
            // 
            // label2
            // 
            label2.Anchor = AnchorStyles.None;
            label2.AutoSize = true;
            label2.BackColor = Color.Transparent;
            label2.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label2.ForeColor = Color.Black;
            label2.Image = (Image)resources.GetObject("label2.Image");
            label2.ImageAlign = ContentAlignment.MiddleLeft;
            label2.Location = new Point(180, 135);
            label2.Name = "label2";
            label2.Size = new Size(67, 20);
            label2.TabIndex = 25;
            label2.Text = "      Email";
            label2.TextAlign = ContentAlignment.MiddleRight;
            // 
            // cbsq
            // 
            cbsq.AllowDrop = true;
            cbsq.Anchor = AnchorStyles.None;
            cbsq.BackColor = SystemColors.ControlLight;
            cbsq.Cursor = Cursors.Hand;
            cbsq.DropDownStyle = ComboBoxStyle.DropDownList;
            cbsq.FormattingEnabled = true;
            cbsq.Items.AddRange(new object[] { "What is your favourite color?", "What is your birth place?", "What is your hobby?", "What is your crush name?", "What is your best friend name?" });
            cbsq.Location = new Point(257, 248);
            cbsq.MaxDropDownItems = 5;
            cbsq.Name = "cbsq";
            cbsq.Size = new Size(200, 23);
            cbsq.TabIndex = 27;
            // 
            // lsq
            // 
            lsq.Anchor = AnchorStyles.None;
            lsq.AutoSize = true;
            lsq.BackColor = Color.Transparent;
            lsq.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lsq.ForeColor = Color.Black;
            lsq.Image = (Image)resources.GetObject("lsq.Image");
            lsq.ImageAlign = ContentAlignment.MiddleLeft;
            lsq.Location = new Point(108, 248);
            lsq.Name = "lsq";
            lsq.Size = new Size(140, 20);
            lsq.TabIndex = 28;
            lsq.Text = "     Security Question";
            lsq.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lsqa
            // 
            lsqa.Anchor = AnchorStyles.None;
            lsqa.AutoSize = true;
            lsqa.BackColor = Color.Transparent;
            lsqa.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lsqa.ForeColor = Color.Black;
            lsqa.Image = (Image)resources.GetObject("lsqa.Image");
            lsqa.ImageAlign = ContentAlignment.MiddleLeft;
            lsqa.Location = new Point(108, 287);
            lsqa.Name = "lsqa";
            lsqa.Size = new Size(144, 20);
            lsqa.TabIndex = 30;
            lsqa.Text = "     Security Que. Ans.";
            lsqa.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbsqa
            // 
            tbsqa.Anchor = AnchorStyles.None;
            tbsqa.BackColor = SystemColors.ControlLight;
            tbsqa.Cursor = Cursors.IBeam;
            tbsqa.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbsqa.Location = new Point(257, 288);
            tbsqa.MaxLength = 25;
            tbsqa.Name = "tbsqa";
            tbsqa.PlaceholderText = "Your Answer";
            tbsqa.ScrollBars = ScrollBars.Horizontal;
            tbsqa.Size = new Size(199, 23);
            tbsqa.TabIndex = 29;
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
            lcp1.Location = new Point(257, 328);
            lcp1.Name = "lcp1";
            lcp1.Size = new Size(2, 23);
            lcp1.TabIndex = 31;
            lcp1.TextAlign = ContentAlignment.MiddleCenter;
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
            lcpplus.Location = new Point(295, 328);
            lcpplus.Name = "lcpplus";
            lcpplus.Size = new Size(23, 23);
            lcpplus.TabIndex = 32;
            lcpplus.Text = "+";
            lcpplus.TextAlign = ContentAlignment.MiddleCenter;
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
            lcp2.Location = new Point(323, 328);
            lcp2.Name = "lcp2";
            lcp2.Size = new Size(2, 23);
            lcp2.TabIndex = 33;
            lcp2.TextAlign = ContentAlignment.MiddleCenter;
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
            lcpequal.Location = new Point(363, 328);
            lcpequal.Name = "lcpequal";
            lcpequal.Size = new Size(23, 23);
            lcpequal.TabIndex = 34;
            lcpequal.Text = "=";
            lcpequal.TextAlign = ContentAlignment.MiddleCenter;
            // 
            // tbcans
            // 
            tbcans.Anchor = AnchorStyles.None;
            tbcans.BackColor = SystemColors.ControlLight;
            tbcans.Cursor = Cursors.IBeam;
            tbcans.Font = new Font("Calibri", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            tbcans.Location = new Point(392, 326);
            tbcans.MaxLength = 3;
            tbcans.Name = "tbcans";
            tbcans.PlaceholderText = "?";
            tbcans.ScrollBars = ScrollBars.Horizontal;
            tbcans.Size = new Size(42, 26);
            tbcans.TabIndex = 35;
            // 
            // lcaptcha
            // 
            lcaptcha.Anchor = AnchorStyles.None;
            lcaptcha.AutoSize = true;
            lcaptcha.BackColor = Color.Transparent;
            lcaptcha.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcaptcha.ForeColor = Color.Black;
            lcaptcha.Image = (Image)resources.GetObject("lcaptcha.Image");
            lcaptcha.ImageAlign = ContentAlignment.MiddleLeft;
            lcaptcha.Location = new Point(166, 332);
            lcaptcha.Name = "lcaptcha";
            lcaptcha.Size = new Size(82, 20);
            lcaptcha.TabIndex = 36;
            lcaptcha.Text = "     Captcha";
            lcaptcha.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lconfirmpass
            // 
            lconfirmpass.Anchor = AnchorStyles.None;
            lconfirmpass.AutoSize = true;
            lconfirmpass.BackColor = Color.Transparent;
            lconfirmpass.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lconfirmpass.ForeColor = Color.Black;
            lconfirmpass.Image = (Image)resources.GetObject("lconfirmpass.Image");
            lconfirmpass.ImageAlign = ContentAlignment.MiddleLeft;
            lconfirmpass.Location = new Point(108, 409);
            lconfirmpass.Name = "lconfirmpass";
            lconfirmpass.Size = new Size(144, 20);
            lconfirmpass.TabIndex = 37;
            lconfirmpass.Text = "     Confirm Password";
            lconfirmpass.TextAlign = ContentAlignment.MiddleRight;
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
            bexit.Location = new Point(211, 494);
            bexit.Name = "bexit";
            bexit.Size = new Size(245, 34);
            bexit.TabIndex = 43;
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
            bprevious.Location = new Point(338, 453);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(119, 34);
            bprevious.TabIndex = 42;
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
            bsignup.Location = new Point(211, 453);
            bsignup.Name = "bsignup";
            bsignup.Size = new Size(119, 34);
            bsignup.TabIndex = 41;
            bsignup.Text = "   Signup";
            bsignup.UseVisualStyleBackColor = false;
            bsignup.Click += bsignup_Click;
            // 
            // tbcpass
            // 
            tbcpass.Anchor = AnchorStyles.None;
            tbcpass.BackColor = SystemColors.ControlLight;
            tbcpass.Cursor = Cursors.IBeam;
            tbcpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbcpass.Location = new Point(258, 409);
            tbcpass.MaxLength = 25;
            tbcpass.Name = "tbcpass";
            tbcpass.PlaceholderText = "Retype Your Password";
            tbcpass.ScrollBars = ScrollBars.Horizontal;
            tbcpass.Size = new Size(199, 23);
            tbcpass.TabIndex = 44;
            tbcpass.UseSystemPasswordChar = true;
            // 
            // tbaddress
            // 
            tbaddress.Anchor = AnchorStyles.None;
            tbaddress.BackColor = SystemColors.ControlLight;
            tbaddress.Cursor = Cursors.IBeam;
            tbaddress.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbaddress.Location = new Point(256, 212);
            tbaddress.MaxLength = 150;
            tbaddress.Name = "tbaddress";
            tbaddress.PlaceholderText = "Your Home Adress";
            tbaddress.ScrollBars = ScrollBars.Horizontal;
            tbaddress.Size = new Size(201, 23);
            tbaddress.TabIndex = 46;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.BackColor = Color.Transparent;
            label1.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label1.ForeColor = Color.Black;
            label1.Image = (Image)resources.GetObject("label1.Image");
            label1.ImageAlign = ContentAlignment.MiddleLeft;
            label1.Location = new Point(168, 212);
            label1.Name = "label1";
            label1.Size = new Size(84, 20);
            label1.TabIndex = 45;
            label1.Text = "      Address";
            label1.TextAlign = ContentAlignment.MiddleRight;
            // 
            // brecapta
            // 
            brecapta.Anchor = AnchorStyles.None;
            brecapta.BackColor = SystemColors.ScrollBar;
            brecapta.BackgroundImageLayout = ImageLayout.Stretch;
            brecapta.Cursor = Cursors.Hand;
            brecapta.FlatStyle = FlatStyle.Popup;
            brecapta.Image = (Image)resources.GetObject("brecapta.Image");
            brecapta.Location = new Point(432, 326);
            brecapta.Name = "brecapta";
            brecapta.Size = new Size(25, 26);
            brecapta.TabIndex = 82;
            brecapta.UseMnemonic = false;
            brecapta.UseVisualStyleBackColor = false;
            brecapta.Click += brecapta_Click;
            // 
            // CustomerSignup
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.ControlLight;
            BackgroundImage = (Image)resources.GetObject("$this.BackgroundImage");
            BackgroundImageLayout = ImageLayout.Stretch;
            ClientSize = new Size(884, 561);
            Controls.Add(brecapta);
            Controls.Add(tbaddress);
            Controls.Add(label1);
            Controls.Add(tbcpass);
            Controls.Add(bexit);
            Controls.Add(bprevious);
            Controls.Add(bsignup);
            Controls.Add(lconfirmpass);
            Controls.Add(lcaptcha);
            Controls.Add(tbcans);
            Controls.Add(lcpequal);
            Controls.Add(lcp2);
            Controls.Add(lcpplus);
            Controls.Add(lcp1);
            Controls.Add(lsqa);
            Controls.Add(tbsqa);
            Controls.Add(lsq);
            Controls.Add(cbsq);
            Controls.Add(tbemail);
            Controls.Add(label2);
            Controls.Add(lbdate);
            Controls.Add(birthdaypicker);
            Controls.Add(tbname);
            Controls.Add(lname);
            Controls.Add(tbpass);
            Controls.Add(tbusername);
            Controls.Add(lpass);
            Controls.Add(lusername);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "CustomerSignup";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "CustomerSignup";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button beyeok;
        private Button beyecross;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label lpass;
        private Label lusername;
        private TextBox tbname;
        private Label lname;
        private DateTimePicker birthdaypicker;
        private Label lbdate;
        private TextBox tbemail;
        private Label label2;
        private ComboBox cbsq;
        private Label lsq;
        private Label lsqa;
        private TextBox tbsqa;
        private Label lcp1;
        private Label lcpplus;
        private Label lcp2;
        private Label lcpequal;
        private TextBox tbcans;
        private Label lcaptcha;
        private Button button1;
        private Button button2;
        private TextBox textBox1;
        private Label lconfirmpass;
        private Button bexit;
        private Button bprevious;
        private Button bsignup;
        private TextBox tbcpass;
        private TextBox tbaddress;
        private Label label1;
        private Button brecapta;
    }
}