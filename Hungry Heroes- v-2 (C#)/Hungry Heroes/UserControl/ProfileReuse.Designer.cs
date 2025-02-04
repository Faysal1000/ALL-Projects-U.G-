namespace Hungry_Heroes
{
    partial class ProfileReuse
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

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ProfileReuse));
            blogout = new Button();
            lprofiletitle = new Label();
            lsetname = new Label();
            label2 = new Label();
            label3 = new Label();
            lbdate = new Label();
            lname = new Label();
            lusername = new Label();
            lsetusername = new Label();
            lsetemail = new Label();
            lsetbirth = new Label();
            lsetaddress = new Label();
            bupdate = new Button();
            SuspendLayout();
            // 
            // blogout
            // 
            blogout.Anchor = AnchorStyles.Right;
            blogout.BackColor = Color.Transparent;
            blogout.Cursor = Cursors.Hand;
            blogout.FlatAppearance.BorderSize = 3;
            blogout.FlatAppearance.MouseDownBackColor = Color.Gray;
            blogout.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            blogout.FlatStyle = FlatStyle.Popup;
            blogout.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            blogout.ForeColor = Color.Black;
            blogout.Image = Properties.Resources.exiticon;
            blogout.ImageAlign = ContentAlignment.MiddleLeft;
            blogout.Location = new Point(503, 360);
            blogout.Name = "blogout";
            blogout.Size = new Size(165, 42);
            blogout.TabIndex = 16;
            blogout.Text = "Logout";
            blogout.UseVisualStyleBackColor = false;
            blogout.Click += blogout_Click;
            // 
            // lprofiletitle
            // 
            lprofiletitle.Anchor = AnchorStyles.None;
            lprofiletitle.AutoSize = true;
            lprofiletitle.BackColor = Color.Transparent;
            lprofiletitle.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Underline, GraphicsUnit.Point);
            lprofiletitle.Location = new Point(289, 33);
            lprofiletitle.Name = "lprofiletitle";
            lprofiletitle.Size = new Size(121, 22);
            lprofiletitle.TabIndex = 15;
            lprofiletitle.Text = "User Details";
            // 
            // lsetname
            // 
            lsetname.AutoSize = true;
            lsetname.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            lsetname.Location = new Point(178, 107);
            lsetname.Name = "lsetname";
            lsetname.Size = new Size(47, 15);
            lsetname.TabIndex = 17;
            lsetname.Text = "label1";
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
            label2.Location = new Point(77, 260);
            label2.Name = "label2";
            label2.Size = new Size(88, 20);
            label2.TabIndex = 50;
            label2.Text = "      Address:";
            label2.TextAlign = ContentAlignment.MiddleRight;
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
            label3.Location = new Point(89, 183);
            label3.Name = "label3";
            label3.Size = new Size(71, 20);
            label3.TabIndex = 49;
            label3.Text = "      Email:";
            label3.TextAlign = ContentAlignment.MiddleRight;
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
            lbdate.Location = new Point(61, 221);
            lbdate.Name = "lbdate";
            lbdate.Size = new Size(102, 20);
            lbdate.TabIndex = 48;
            lbdate.Text = "      Birth-Date:";
            lbdate.TextAlign = ContentAlignment.MiddleRight;
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
            lname.Location = new Point(89, 103);
            lname.Name = "lname";
            lname.Size = new Size(75, 20);
            lname.TabIndex = 47;
            lname.Text = "      Name:";
            lname.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lusername
            // 
            lusername.Anchor = AnchorStyles.None;
            lusername.AutoSize = true;
            lusername.BackColor = Color.Transparent;
            lusername.Font = new Font("Franklin Gothic Book", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lusername.ForeColor = Color.Black;
            lusername.Image = (Image)resources.GetObject("lusername.Image");
            lusername.ImageAlign = ContentAlignment.MiddleLeft;
            lusername.Location = new Point(62, 143);
            lusername.Name = "lusername";
            lusername.Size = new Size(102, 20);
            lusername.TabIndex = 46;
            lusername.Text = "      Username:";
            lusername.TextAlign = ContentAlignment.MiddleRight;
            // 
            // lsetusername
            // 
            lsetusername.AutoSize = true;
            lsetusername.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            lsetusername.Location = new Point(178, 148);
            lsetusername.Name = "lsetusername";
            lsetusername.Size = new Size(47, 15);
            lsetusername.TabIndex = 51;
            lsetusername.Text = "label1";
            // 
            // lsetemail
            // 
            lsetemail.AutoSize = true;
            lsetemail.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            lsetemail.Location = new Point(178, 188);
            lsetemail.Name = "lsetemail";
            lsetemail.Size = new Size(47, 15);
            lsetemail.TabIndex = 52;
            lsetemail.Text = "label1";
            // 
            // lsetbirth
            // 
            lsetbirth.AutoSize = true;
            lsetbirth.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            lsetbirth.Location = new Point(178, 226);
            lsetbirth.Name = "lsetbirth";
            lsetbirth.Size = new Size(47, 15);
            lsetbirth.TabIndex = 53;
            lsetbirth.Text = "label1";
            // 
            // lsetaddress
            // 
            lsetaddress.AutoSize = true;
            lsetaddress.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            lsetaddress.Location = new Point(178, 265);
            lsetaddress.Name = "lsetaddress";
            lsetaddress.Size = new Size(47, 15);
            lsetaddress.TabIndex = 54;
            lsetaddress.Text = "label1";
            // 
            // bupdate
            // 
            bupdate.Anchor = AnchorStyles.Right;
            bupdate.BackColor = Color.Transparent;
            bupdate.Cursor = Cursors.Hand;
            bupdate.FlatAppearance.BorderSize = 3;
            bupdate.FlatAppearance.MouseDownBackColor = Color.Gray;
            bupdate.FlatAppearance.MouseOverBackColor = Color.FromArgb(64, 64, 64);
            bupdate.FlatStyle = FlatStyle.Popup;
            bupdate.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bupdate.ForeColor = Color.Black;
            bupdate.Image = (Image)resources.GetObject("bupdate.Image");
            bupdate.ImageAlign = ContentAlignment.MiddleLeft;
            bupdate.Location = new Point(336, 360);
            bupdate.Name = "bupdate";
            bupdate.Size = new Size(161, 42);
            bupdate.TabIndex = 55;
            bupdate.Text = "      Update Details";
            bupdate.UseVisualStyleBackColor = false;
            bupdate.Click += bupdate_Click;
            // 
            // ProfileReuse
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            Controls.Add(bupdate);
            Controls.Add(lsetaddress);
            Controls.Add(lsetbirth);
            Controls.Add(lsetemail);
            Controls.Add(lsetusername);
            Controls.Add(label2);
            Controls.Add(label3);
            Controls.Add(lbdate);
            Controls.Add(lname);
            Controls.Add(lusername);
            Controls.Add(lsetname);
            Controls.Add(blogout);
            Controls.Add(lprofiletitle);
            Name = "ProfileReuse";
            Size = new Size(745, 476);
            Load += ProfileReuse_Load;
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Button blogout;
        private Label lprofiletitle;
        private Label lsetname;
        private Label label2;
        private Label label3;
        private Label lbdate;
        private Label lname;
        private Label lusername;
        private Label lsetusername;
        private Label lsetemail;
        private Label lsetbirth;
        private Label lsetaddress;
        private Button bupdate;
    }
}
