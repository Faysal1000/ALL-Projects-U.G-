namespace Hungry_Heroes
{
    partial class CustomerEdit
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CustomerEdit));
            tbaddress = new TextBox();
            label4 = new Label();
            lbdate = new Label();
            birthdaypicker = new DateTimePicker();
            ltitle = new Label();
            bupdate = new Button();
            bcancel = new Button();
            lsqa = new Label();
            tbsqa = new TextBox();
            lsq = new Label();
            cbsq = new ComboBox();
            tbemail = new TextBox();
            label2 = new Label();
            tbname = new TextBox();
            lname = new Label();
            tbpass = new TextBox();
            tbusername = new TextBox();
            lpass = new Label();
            lusername = new Label();
            SuspendLayout();
            // 
            // tbaddress
            // 
            tbaddress.Anchor = AnchorStyles.None;
            tbaddress.BackColor = SystemColors.ControlLight;
            tbaddress.Cursor = Cursors.IBeam;
            tbaddress.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbaddress.Location = new Point(151, 187);
            tbaddress.MaxLength = 150;
            tbaddress.Name = "tbaddress";
            tbaddress.PlaceholderText = "Your Home Adress";
            tbaddress.ScrollBars = ScrollBars.Horizontal;
            tbaddress.Size = new Size(201, 23);
            tbaddress.TabIndex = 73;
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
            label4.Location = new Point(63, 187);
            label4.Name = "label4";
            label4.Size = new Size(84, 20);
            label4.TabIndex = 72;
            label4.Text = "      Address";
            label4.TextAlign = ContentAlignment.MiddleRight;
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
            lbdate.Location = new Point(47, 158);
            lbdate.Name = "lbdate";
            lbdate.Size = new Size(98, 20);
            lbdate.TabIndex = 71;
            lbdate.Text = "      Birth-Date";
            lbdate.TextAlign = ContentAlignment.MiddleRight;
            // 
            // birthdaypicker
            // 
            birthdaypicker.Anchor = AnchorStyles.None;
            birthdaypicker.Cursor = Cursors.Hand;
            birthdaypicker.Location = new Point(151, 158);
            birthdaypicker.MaxDate = new DateTime(2029, 7, 5, 0, 0, 0, 0);
            birthdaypicker.Name = "birthdaypicker";
            birthdaypicker.Size = new Size(200, 23);
            birthdaypicker.TabIndex = 70;
            // 
            // ltitle
            // 
            ltitle.Anchor = AnchorStyles.None;
            ltitle.AutoSize = true;
            ltitle.BackColor = Color.Transparent;
            ltitle.Font = new Font("Britannic Bold", 12F, FontStyle.Underline, GraphicsUnit.Point);
            ltitle.ForeColor = Color.Black;
            ltitle.ImageAlign = ContentAlignment.MiddleLeft;
            ltitle.Location = new Point(119, 24);
            ltitle.Name = "ltitle";
            ltitle.Size = new Size(184, 17);
            ltitle.TabIndex = 69;
            ltitle.Text = "Update Customer Details";
            ltitle.TextAlign = ContentAlignment.MiddleRight;
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
            bupdate.Location = new Point(152, 334);
            bupdate.Name = "bupdate";
            bupdate.Size = new Size(96, 27);
            bupdate.TabIndex = 68;
            bupdate.Text = "    Update";
            bupdate.UseVisualStyleBackColor = false;
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
            bcancel.Location = new Point(254, 334);
            bcancel.Name = "bcancel";
            bcancel.Size = new Size(97, 27);
            bcancel.TabIndex = 67;
            bcancel.Text = "    Cancel";
            bcancel.UseVisualStyleBackColor = false;
            bcancel.Click += bcancel_Click;
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
            lsqa.Location = new Point(2, 244);
            lsqa.Name = "lsqa";
            lsqa.Size = new Size(144, 20);
            lsqa.TabIndex = 64;
            lsqa.Text = "     Security Que. Ans.";
            lsqa.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbsqa
            // 
            tbsqa.Anchor = AnchorStyles.None;
            tbsqa.BackColor = SystemColors.ControlLight;
            tbsqa.Cursor = Cursors.IBeam;
            tbsqa.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbsqa.Location = new Point(151, 245);
            tbsqa.MaxLength = 25;
            tbsqa.Name = "tbsqa";
            tbsqa.PlaceholderText = "Answer";
            tbsqa.ScrollBars = ScrollBars.Horizontal;
            tbsqa.Size = new Size(199, 23);
            tbsqa.TabIndex = 63;
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
            lsq.Location = new Point(2, 216);
            lsq.Name = "lsq";
            lsq.Size = new Size(140, 20);
            lsq.TabIndex = 62;
            lsq.Text = "     Security Question";
            lsq.TextAlign = ContentAlignment.MiddleRight;
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
            cbsq.Location = new Point(151, 216);
            cbsq.MaxDropDownItems = 5;
            cbsq.Name = "cbsq";
            cbsq.Size = new Size(200, 23);
            cbsq.TabIndex = 61;
            // 
            // tbemail
            // 
            tbemail.Anchor = AnchorStyles.None;
            tbemail.BackColor = SystemColors.ControlLight;
            tbemail.Cursor = Cursors.IBeam;
            tbemail.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbemail.Location = new Point(151, 129);
            tbemail.MaxLength = 40;
            tbemail.Name = "tbemail";
            tbemail.PlaceholderText = "Email Adress";
            tbemail.ScrollBars = ScrollBars.Horizontal;
            tbemail.Size = new Size(199, 23);
            tbemail.TabIndex = 60;
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
            label2.Location = new Point(74, 129);
            label2.Name = "label2";
            label2.Size = new Size(67, 20);
            label2.TabIndex = 59;
            label2.Text = "      Email";
            label2.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.ControlLight;
            tbname.Cursor = Cursors.IBeam;
            tbname.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbname.Location = new Point(151, 71);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Full Name";
            tbname.ScrollBars = ScrollBars.Horizontal;
            tbname.Size = new Size(199, 23);
            tbname.TabIndex = 55;
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
            lname.Location = new Point(74, 71);
            lname.Name = "lname";
            lname.Size = new Size(71, 20);
            lname.TabIndex = 57;
            lname.Text = "      Name";
            lname.TextAlign = ContentAlignment.MiddleRight;
            // 
            // tbpass
            // 
            tbpass.Anchor = AnchorStyles.None;
            tbpass.BackColor = SystemColors.ControlLight;
            tbpass.Cursor = Cursors.IBeam;
            tbpass.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbpass.Location = new Point(151, 274);
            tbpass.MaxLength = 25;
            tbpass.Name = "tbpass";
            tbpass.PlaceholderText = "Password";
            tbpass.ScrollBars = ScrollBars.Horizontal;
            tbpass.Size = new Size(199, 23);
            tbpass.TabIndex = 56;
            // 
            // tbusername
            // 
            tbusername.Anchor = AnchorStyles.None;
            tbusername.BackColor = SystemColors.ControlLight;
            tbusername.Cursor = Cursors.IBeam;
            tbusername.Font = new Font("Calibri", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            tbusername.Location = new Point(151, 100);
            tbusername.MaxLength = 25;
            tbusername.Name = "tbusername";
            tbusername.PlaceholderText = "Username";
            tbusername.ScrollBars = ScrollBars.Horizontal;
            tbusername.Size = new Size(199, 23);
            tbusername.TabIndex = 58;
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
            lpass.Location = new Point(55, 275);
            lpass.Name = "lpass";
            lpass.Size = new Size(90, 20);
            lpass.TabIndex = 54;
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
            lusername.Location = new Point(47, 100);
            lusername.Name = "lusername";
            lusername.Size = new Size(98, 20);
            lusername.TabIndex = 53;
            lusername.Text = "      Username";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // CustomerEdit
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(412, 384);
            Controls.Add(tbaddress);
            Controls.Add(label4);
            Controls.Add(lbdate);
            Controls.Add(birthdaypicker);
            Controls.Add(ltitle);
            Controls.Add(bupdate);
            Controls.Add(bcancel);
            Controls.Add(lsqa);
            Controls.Add(tbsqa);
            Controls.Add(lsq);
            Controls.Add(cbsq);
            Controls.Add(tbemail);
            Controls.Add(label2);
            Controls.Add(tbname);
            Controls.Add(lname);
            Controls.Add(tbpass);
            Controls.Add(tbusername);
            Controls.Add(lpass);
            Controls.Add(lusername);
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "CustomerEdit";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "CustomerEdit";
            Load += CustomerEdit_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private TextBox tbaddress;
        private Label label4;
        private Label lbdate;
        private DateTimePicker birthdaypicker;
        private Label ltitle;
        private Button bupdate;
        private Button bcancel;
        private Label lsqa;
        private TextBox tbsqa;
        private Label lsq;
        private ComboBox cbsq;
        private TextBox tbemail;
        private Label label2;
        private TextBox tbname;
        private Label lname;
        private TextBox tbpass;
        private TextBox tbusername;
        private Label lpass;
        private Label lusername;
    }
}