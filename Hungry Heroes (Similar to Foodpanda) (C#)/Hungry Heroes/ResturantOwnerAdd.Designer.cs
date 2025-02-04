namespace Hungry_Heroes
{
    partial class ResturantOwnerAdd
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ResturantOwnerAdd));
            badd = new Button();
            bupdate = new Button();
            bcancel = new Button();
            tbphone = new TextBox();
            lcontact = new Label();
            tbresname = new TextBox();
            lresturantname = new Label();
            tbemail = new TextBox();
            tbname = new TextBox();
            tbpass = new TextBox();
            tbusername = new TextBox();
            label2 = new Label();
            lname = new Label();
            lusername = new Label();
            lpass = new Label();
            label1 = new Label();
            label3 = new Label();
            panel1 = new Panel();
            rbdeactive = new RadioButton();
            rbactive = new RadioButton();
            panel2 = new Panel();
            no = new RadioButton();
            yes = new RadioButton();
            ltitle = new Label();
            label4 = new Label();
            tbdiscount = new TextBox();
            panel1.SuspendLayout();
            panel2.SuspendLayout();
            SuspendLayout();
            // 
            // badd
            // 
            badd.Anchor = AnchorStyles.None;
            badd.BackColor = Color.Transparent;
            badd.Cursor = Cursors.Hand;
            badd.FlatAppearance.BorderColor = Color.FromArgb(0, 192, 0);
            badd.FlatAppearance.MouseDownBackColor = Color.Red;
            badd.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            badd.FlatStyle = FlatStyle.Flat;
            badd.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            badd.Image = (Image)resources.GetObject("badd.Image");
            badd.ImageAlign = ContentAlignment.MiddleLeft;
            badd.Location = new Point(153, 345);
            badd.Name = "badd";
            badd.Size = new Size(98, 27);
            badd.TabIndex = 55;
            badd.Text = "      Add User";
            badd.UseVisualStyleBackColor = false;
            badd.Visible = false;
            badd.Click += badd_Click;
            // 
            // bupdate
            // 
            bupdate.Anchor = AnchorStyles.None;
            bupdate.BackColor = Color.Transparent;
            bupdate.Cursor = Cursors.Hand;
            bupdate.FlatAppearance.BorderColor = Color.FromArgb(0, 192, 0);
            bupdate.FlatAppearance.MouseDownBackColor = Color.Red;
            bupdate.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            bupdate.FlatStyle = FlatStyle.Flat;
            bupdate.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            bupdate.Image = (Image)resources.GetObject("bupdate.Image");
            bupdate.ImageAlign = ContentAlignment.MiddleLeft;
            bupdate.Location = new Point(153, 345);
            bupdate.Name = "bupdate";
            bupdate.Size = new Size(96, 27);
            bupdate.TabIndex = 54;
            bupdate.Text = "    Update";
            bupdate.UseVisualStyleBackColor = false;
            bupdate.Visible = false;
            bupdate.Click += bupdate_Click;
            // 
            // bcancel
            // 
            bcancel.Anchor = AnchorStyles.None;
            bcancel.BackColor = Color.Transparent;
            bcancel.Cursor = Cursors.Hand;
            bcancel.FlatAppearance.BorderColor = Color.Red;
            bcancel.FlatAppearance.MouseDownBackColor = Color.Red;
            bcancel.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            bcancel.FlatStyle = FlatStyle.Flat;
            bcancel.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            bcancel.Image = Properties.Resources.close;
            bcancel.ImageAlign = ContentAlignment.MiddleLeft;
            bcancel.Location = new Point(255, 345);
            bcancel.Name = "bcancel";
            bcancel.Size = new Size(97, 27);
            bcancel.TabIndex = 53;
            bcancel.Text = "    Cancel";
            bcancel.UseVisualStyleBackColor = false;
            bcancel.Click += bcancel_Click;
            // 
            // tbphone
            // 
            tbphone.Anchor = AnchorStyles.None;
            tbphone.BackColor = SystemColors.ControlLight;
            tbphone.Cursor = Cursors.IBeam;
            tbphone.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbphone.Location = new Point(154, 151);
            tbphone.MaxLength = 11;
            tbphone.Name = "tbphone";
            tbphone.PlaceholderText = "Phone Number";
            tbphone.ScrollBars = ScrollBars.Horizontal;
            tbphone.Size = new Size(199, 23);
            tbphone.TabIndex = 79;
            // 
            // lcontact
            // 
            lcontact.Anchor = AnchorStyles.None;
            lcontact.AutoSize = true;
            lcontact.BackColor = Color.Transparent;
            lcontact.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lcontact.ForeColor = SystemColors.ActiveCaptionText;
            lcontact.Image = (Image)resources.GetObject("lcontact.Image");
            lcontact.ImageAlign = ContentAlignment.MiddleLeft;
            lcontact.Location = new Point(27, 154);
            lcontact.Name = "lcontact";
            lcontact.Size = new Size(121, 20);
            lcontact.TabIndex = 78;
            lcontact.Text = "          Contact No";
            lcontact.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbresname
            // 
            tbresname.Anchor = AnchorStyles.None;
            tbresname.BackColor = SystemColors.ControlLight;
            tbresname.Cursor = Cursors.IBeam;
            tbresname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbresname.Location = new Point(154, 184);
            tbresname.MaxLength = 25;
            tbresname.Name = "tbresname";
            tbresname.PlaceholderText = "Your Resturant Name";
            tbresname.ScrollBars = ScrollBars.Horizontal;
            tbresname.Size = new Size(199, 23);
            tbresname.TabIndex = 77;
            // 
            // lresturantname
            // 
            lresturantname.Anchor = AnchorStyles.None;
            lresturantname.AutoSize = true;
            lresturantname.BackColor = Color.Transparent;
            lresturantname.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lresturantname.ForeColor = SystemColors.ActiveCaptionText;
            lresturantname.Image = (Image)resources.GetObject("lresturantname.Image");
            lresturantname.ImageAlign = ContentAlignment.MiddleLeft;
            lresturantname.Location = new Point(8, 184);
            lresturantname.Name = "lresturantname";
            lresturantname.Size = new Size(134, 20);
            lresturantname.TabIndex = 76;
            lresturantname.Text = "     Resturnat Name";
            lresturantname.TextAlign = ContentAlignment.MiddleLeft;
            // 
            // tbemail
            // 
            tbemail.Anchor = AnchorStyles.None;
            tbemail.BackColor = SystemColors.ControlLight;
            tbemail.Cursor = Cursors.IBeam;
            tbemail.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbemail.Location = new Point(154, 122);
            tbemail.MaxLength = 40;
            tbemail.Name = "tbemail";
            tbemail.PlaceholderText = "Email Adress";
            tbemail.ScrollBars = ScrollBars.Horizontal;
            tbemail.Size = new Size(199, 23);
            tbemail.TabIndex = 75;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.ControlLight;
            tbname.Cursor = Cursors.IBeam;
            tbname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbname.Location = new Point(154, 64);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Full Name";
            tbname.ScrollBars = ScrollBars.Horizontal;
            tbname.Size = new Size(199, 23);
            tbname.TabIndex = 70;
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ControlLight;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(153, 310);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(199, 23);
            tbpass.TabIndex = 71;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ControlLight;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(154, 93);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 73;
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
            label2.Location = new Point(77, 125);
            label2.Name = "label2";
            label2.Size = new Size(67, 20);
            label2.TabIndex = 82;
            label2.Text = "      Email";
            label2.TextAlign = ContentAlignment.MiddleRight;
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
            lname.Location = new Point(77, 67);
            lname.Name = "lname";
            lname.Size = new Size(71, 20);
            lname.TabIndex = 81;
            lname.Text = "      Name";
            lname.TextAlign = ContentAlignment.MiddleRight;
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
            lusername.Location = new Point(50, 96);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 80;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
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
            lpass.Location = new Point(57, 313);
            lpass.Name = "lpass";
            lpass.Size = new Size(90, 20);
            lpass.TabIndex = 83;
            lpass.Text = "     Password";
            lpass.TextAlign = ContentAlignment.MiddleRight;
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
            label1.Location = new Point(19, 213);
            label1.Name = "label1";
            label1.Size = new Size(126, 20);
            label1.TabIndex = 85;
            label1.Text = "     Account Status";
            label1.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label3
            // 
            label3.Anchor = AnchorStyles.None;
            label3.AutoSize = true;
            label3.BackColor = Color.Transparent;
            label3.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label3.ForeColor = Color.Black;
            label3.Image = (Image)resources.GetObject("label3.Image");
            label3.ImageAlign = ContentAlignment.MiddleLeft;
            label3.Location = new Point(48, 251);
            label3.Name = "label3";
            label3.Size = new Size(94, 20);
            label3.TabIndex = 86;
            label3.Text = "     Availability";
            label3.TextAlign = ContentAlignment.MiddleRight;
            // 
            // panel1
            // 
            panel1.Anchor = AnchorStyles.None;
            panel1.Controls.Add(rbdeactive);
            panel1.Controls.Add(rbactive);
            panel1.Location = new Point(153, 211);
            panel1.Name = "panel1";
            panel1.Size = new Size(200, 28);
            panel1.TabIndex = 89;
            // 
            // rbdeactive
            // 
            rbdeactive.AutoSize = true;
            rbdeactive.Location = new Point(67, 3);
            rbdeactive.Name = "rbdeactive";
            rbdeactive.Size = new Size(70, 19);
            rbdeactive.TabIndex = 90;
            rbdeactive.TabStop = true;
            rbdeactive.Text = "Deactive";
            rbdeactive.UseVisualStyleBackColor = true;
            // 
            // rbactive
            // 
            rbactive.AutoSize = true;
            rbactive.Location = new Point(3, 3);
            rbactive.Name = "rbactive";
            rbactive.Size = new Size(58, 19);
            rbactive.TabIndex = 89;
            rbactive.TabStop = true;
            rbactive.Text = "Active";
            rbactive.UseVisualStyleBackColor = true;
            // 
            // panel2
            // 
            panel2.Anchor = AnchorStyles.None;
            panel2.Controls.Add(no);
            panel2.Controls.Add(yes);
            panel2.Location = new Point(153, 245);
            panel2.Name = "panel2";
            panel2.Size = new Size(200, 32);
            panel2.TabIndex = 90;
            // 
            // no
            // 
            no.AutoSize = true;
            no.Location = new Point(52, 6);
            no.Name = "no";
            no.Size = new Size(41, 19);
            no.TabIndex = 90;
            no.TabStop = true;
            no.Text = "No";
            no.UseVisualStyleBackColor = true;
            // 
            // yes
            // 
            yes.AutoSize = true;
            yes.Location = new Point(4, 6);
            yes.Name = "yes";
            yes.Size = new Size(42, 19);
            yes.TabIndex = 89;
            yes.TabStop = true;
            yes.Text = "Yes";
            yes.UseVisualStyleBackColor = true;
            // 
            // ltitle
            // 
            ltitle.Anchor = AnchorStyles.None;
            ltitle.AutoSize = true;
            ltitle.BackColor = Color.Transparent;
            ltitle.Font = new Font("Britannic Bold", 12F, FontStyle.Underline, GraphicsUnit.Point);
            ltitle.ForeColor = Color.Black;
            ltitle.ImageAlign = ContentAlignment.MiddleLeft;
            ltitle.Location = new Point(119, 23);
            ltitle.Name = "ltitle";
            ltitle.Size = new Size(234, 17);
            ltitle.TabIndex = 91;
            ltitle.Text = "Update Resturant Owner Details";
            ltitle.TextAlign = ContentAlignment.MiddleRight;
            // 
            // label4
            // 
            label4.Anchor = AnchorStyles.None;
            label4.AutoSize = true;
            label4.BackColor = Color.Transparent;
            label4.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label4.ForeColor = Color.Black;
            label4.Image = (Image)resources.GetObject("label4.Image");
            label4.ImageAlign = ContentAlignment.MiddleLeft;
            label4.Location = new Point(57, 285);
            label4.Name = "label4";
            label4.Size = new Size(85, 20);
            label4.TabIndex = 93;
            label4.Text = "     Discount";
            label4.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbdiscount
            // 
            tbdiscount.Anchor = AnchorStyles.None;
            tbdiscount.BackColor = SystemColors.ControlLight;
            tbdiscount.Cursor = Cursors.IBeam;
            tbdiscount.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbdiscount.Location = new Point(153, 282);
            tbdiscount.MaxLength = 25;
            tbdiscount.Name = "tbdiscount";
            tbdiscount.PlaceholderText = "Discount";
            tbdiscount.ScrollBars = ScrollBars.Horizontal;
            tbdiscount.Size = new Size(199, 23);
            tbdiscount.TabIndex = 92;
            // 
            // ResturantOwnerAdd
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(413, 409);
            Controls.Add(label4);
            Controls.Add(tbdiscount);
            Controls.Add(ltitle);
            Controls.Add(panel2);
            Controls.Add(panel1);
            Controls.Add(label3);
            Controls.Add(label1);
            Controls.Add(lpass);
            Controls.Add(label2);
            Controls.Add(lname);
            Controls.Add(lusername);
            Controls.Add(tbphone);
            Controls.Add(lcontact);
            Controls.Add(tbresname);
            Controls.Add(lresturantname);
            Controls.Add(tbemail);
            Controls.Add(tbname);
            Controls.Add(tbpass);
            Controls.Add(tbusername);
            Controls.Add(badd);
            Controls.Add(bupdate);
            Controls.Add(bcancel);
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "ResturantOwnerAdd";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "ResturantOwnerAdd";
            Load += ResturantOwnerAdd_Load;
            panel1.ResumeLayout(false);
            panel1.PerformLayout();
            panel2.ResumeLayout(false);
            panel2.PerformLayout();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button badd;
        private Button bupdate;
        private Button bcancel;
        private TextBox tbphone;
        private Label lcontact;
        private TextBox tbresname;
        private Label lresturantname;
        private TextBox tbemail;
        private TextBox tbname;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label label2;
        private Label lname;
        private Label lusername;
        private Label lpass;
        private Label label1;
        private Label label3;
        private Panel panel1;
        private RadioButton rbdeactive;
        private RadioButton rbactive;
        private Panel panel2;
        private RadioButton no;
        private RadioButton yes;
        private Label ltitle;
        private Label label4;
        private TextBox tbdiscount;
    }
}