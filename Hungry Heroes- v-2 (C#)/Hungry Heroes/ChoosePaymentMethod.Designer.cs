namespace Hungry_Heroes
{
    partial class ChoosePaymentMethod
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ChoosePaymentMethod));
            DataGridViewCellStyle dataGridViewCellStyle3 = new DataGridViewCellStyle();
            DataGridViewCellStyle dataGridViewCellStyle4 = new DataGridViewCellStyle();
            label1 = new Label();
            bbkash = new Button();
            bmastercard = new Button();
            bclose = new Button();
            laddress = new Label();
            ptb1 = new Panel();
            lname = new Label();
            tbname = new TextBox();
            ptb2 = new Panel();
            tbaddress = new TextBox();
            dataGridView = new DataGridView();
            ldiscount = new Label();
            use = new CheckBox();
            bdelete = new Button();
            ((System.ComponentModel.ISupportInitialize)dataGridView).BeginInit();
            SuspendLayout();
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Underline, GraphicsUnit.Point);
            label1.Location = new Point(130, 265);
            label1.Name = "label1";
            label1.Size = new Size(200, 15);
            label1.TabIndex = 1;
            label1.Text = "Choose Your Payment Method";
            // 
            // bbkash
            // 
            bbkash.Anchor = AnchorStyles.None;
            bbkash.BackColor = Color.Transparent;
            bbkash.Cursor = Cursors.Hand;
            bbkash.FlatStyle = FlatStyle.Flat;
            bbkash.ForeColor = Color.Transparent;
            bbkash.Image = Properties.Resources.bkash;
            bbkash.Location = new Point(85, 290);
            bbkash.Name = "bbkash";
            bbkash.Size = new Size(137, 87);
            bbkash.TabIndex = 2;
            bbkash.UseVisualStyleBackColor = false;
            bbkash.Click += bbkash_Click;
            // 
            // bmastercard
            // 
            bmastercard.Anchor = AnchorStyles.None;
            bmastercard.BackColor = Color.Transparent;
            bmastercard.Cursor = Cursors.Hand;
            bmastercard.FlatStyle = FlatStyle.Flat;
            bmastercard.ForeColor = Color.Transparent;
            bmastercard.Image = (Image)resources.GetObject("bmastercard.Image");
            bmastercard.Location = new Point(248, 290);
            bmastercard.Name = "bmastercard";
            bmastercard.Size = new Size(137, 87);
            bmastercard.TabIndex = 3;
            bmastercard.UseVisualStyleBackColor = false;
            bmastercard.Click += bmastercard_Click;
            // 
            // bclose
            // 
            bclose.Anchor = AnchorStyles.None;
            bclose.BackColor = Color.Transparent;
            bclose.Cursor = Cursors.Hand;
            bclose.FlatAppearance.BorderColor = Color.Red;
            bclose.FlatAppearance.MouseDownBackColor = Color.Red;
            bclose.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            bclose.FlatStyle = FlatStyle.Flat;
            bclose.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            bclose.Image = Properties.Resources.close;
            bclose.ImageAlign = ContentAlignment.MiddleLeft;
            bclose.Location = new Point(237, 405);
            bclose.Name = "bclose";
            bclose.Size = new Size(121, 27);
            bclose.TabIndex = 4;
            bclose.Text = "    Close";
            bclose.UseVisualStyleBackColor = false;
            bclose.Click += bclose_Click;
            // 
            // laddress
            // 
            laddress.Anchor = AnchorStyles.None;
            laddress.AutoSize = true;
            laddress.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            laddress.Location = new Point(246, 205);
            laddress.Name = "laddress";
            laddress.Size = new Size(153, 14);
            laddress.TabIndex = 94;
            laddress.Text = "Your Address (Required)";
            laddress.Visible = false;
            // 
            // ptb1
            // 
            ptb1.Anchor = AnchorStyles.None;
            ptb1.BackColor = Color.LightCoral;
            ptb1.Location = new Point(13, 235);
            ptb1.Name = "ptb1";
            ptb1.Size = new Size(200, 2);
            ptb1.TabIndex = 92;
            ptb1.Visible = false;
            // 
            // lname
            // 
            lname.Anchor = AnchorStyles.None;
            lname.AutoSize = true;
            lname.Font = new Font("Arial Rounded MT Bold", 9F, FontStyle.Regular, GraphicsUnit.Point);
            lname.Location = new Point(12, 205);
            lname.Name = "lname";
            lname.Size = new Size(136, 14);
            lname.TabIndex = 91;
            lname.Text = "Your Name (Required)";
            lname.Visible = false;
            // 
            // tbname
            // 
            tbname.Anchor = AnchorStyles.None;
            tbname.BackColor = SystemColors.Control;
            tbname.BorderStyle = BorderStyle.None;
            tbname.Cursor = Cursors.IBeam;
            tbname.Location = new Point(16, 219);
            tbname.MaxLength = 25;
            tbname.Name = "tbname";
            tbname.PlaceholderText = "Enter Your Name";
            tbname.Size = new Size(200, 16);
            tbname.TabIndex = 90;
            tbname.Visible = false;
            // 
            // ptb2
            // 
            ptb2.Anchor = AnchorStyles.None;
            ptb2.BackColor = Color.LightCoral;
            ptb2.Location = new Point(249, 238);
            ptb2.Name = "ptb2";
            ptb2.Size = new Size(200, 2);
            ptb2.TabIndex = 96;
            ptb2.Visible = false;
            // 
            // tbaddress
            // 
            tbaddress.Anchor = AnchorStyles.None;
            tbaddress.BackColor = SystemColors.Control;
            tbaddress.BorderStyle = BorderStyle.None;
            tbaddress.Cursor = Cursors.IBeam;
            tbaddress.Location = new Point(250, 222);
            tbaddress.MaxLength = 40;
            tbaddress.Name = "tbaddress";
            tbaddress.PlaceholderText = "Enter Your Address";
            tbaddress.Size = new Size(200, 16);
            tbaddress.TabIndex = 95;
            tbaddress.Visible = false;
            // 
            // dataGridView
            // 
            dataGridView.Anchor = AnchorStyles.None;
            dataGridView.BackgroundColor = SystemColors.ButtonFace;
            dataGridViewCellStyle3.Alignment = DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle3.BackColor = SystemColors.Control;
            dataGridViewCellStyle3.Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
            dataGridViewCellStyle3.ForeColor = SystemColors.WindowText;
            dataGridViewCellStyle3.SelectionBackColor = SystemColors.Highlight;
            dataGridViewCellStyle3.SelectionForeColor = SystemColors.HighlightText;
            dataGridViewCellStyle3.WrapMode = DataGridViewTriState.True;
            dataGridView.ColumnHeadersDefaultCellStyle = dataGridViewCellStyle3;
            dataGridView.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            dataGridViewCellStyle4.Alignment = DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle4.BackColor = SystemColors.Window;
            dataGridViewCellStyle4.Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
            dataGridViewCellStyle4.ForeColor = SystemColors.WindowText;
            dataGridViewCellStyle4.SelectionBackColor = SystemColors.Highlight;
            dataGridViewCellStyle4.SelectionForeColor = SystemColors.HighlightText;
            dataGridViewCellStyle4.WrapMode = DataGridViewTriState.False;
            dataGridView.DefaultCellStyle = dataGridViewCellStyle4;
            dataGridView.GridColor = SystemColors.ActiveCaptionText;
            dataGridView.Location = new Point(4, 3);
            dataGridView.MultiSelect = false;
            dataGridView.Name = "dataGridView";
            dataGridView.RowTemplate.Height = 25;
            dataGridView.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            dataGridView.Size = new Size(476, 191);
            dataGridView.TabIndex = 97;
            // 
            // ldiscount
            // 
            ldiscount.Anchor = AnchorStyles.None;
            ldiscount.AutoSize = true;
            ldiscount.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            ldiscount.ForeColor = Color.Red;
            ldiscount.Location = new Point(74, 220);
            ldiscount.Name = "ldiscount";
            ldiscount.Size = new Size(256, 15);
            ldiscount.TabIndex = 98;
            ldiscount.Text = "15% Discount Available. Wanna Use it?";
            ldiscount.Visible = false;
            // 
            // use
            // 
            use.AutoSize = true;
            use.Location = new Point(336, 218);
            use.Name = "use";
            use.Size = new Size(46, 19);
            use.TabIndex = 99;
            use.Text = "USE";
            use.UseVisualStyleBackColor = true;
            use.Visible = false;
            // 
            // bdelete
            // 
            bdelete.Anchor = AnchorStyles.None;
            bdelete.BackColor = Color.Transparent;
            bdelete.Cursor = Cursors.Hand;
            bdelete.FlatAppearance.BorderColor = Color.Red;
            bdelete.FlatAppearance.MouseDownBackColor = Color.Red;
            bdelete.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            bdelete.FlatStyle = FlatStyle.Flat;
            bdelete.Font = new Font("Arial Rounded MT Bold", 9.75F, FontStyle.Regular, GraphicsUnit.Point);
            bdelete.Image = Properties.Resources.close;
            bdelete.ImageAlign = ContentAlignment.MiddleLeft;
            bdelete.Location = new Point(108, 405);
            bdelete.Name = "bdelete";
            bdelete.Size = new Size(123, 27);
            bdelete.TabIndex = 100;
            bdelete.Text = "    Delete Item";
            bdelete.UseVisualStyleBackColor = false;
            bdelete.Click += button1_Click;
            // 
            // ChoosePaymentMethod
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(484, 474);
            Controls.Add(bdelete);
            Controls.Add(use);
            Controls.Add(ldiscount);
            Controls.Add(dataGridView);
            Controls.Add(ptb2);
            Controls.Add(tbaddress);
            Controls.Add(laddress);
            Controls.Add(ptb1);
            Controls.Add(lname);
            Controls.Add(tbname);
            Controls.Add(bclose);
            Controls.Add(bmastercard);
            Controls.Add(bbkash);
            Controls.Add(label1);
            FormBorderStyle = FormBorderStyle.None;
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "ChoosePaymentMethod";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "ChoosePaymentMethod";
            Load += ChoosePaymentMethod_Load;
            ((System.ComponentModel.ISupportInitialize)dataGridView).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion
        private Label label1;
        private Button bbkash;
        private Button bmastercard;
        private Button bclose;
        private Label laddress;
        private Panel ptb1;
        private Label lname;
        private TextBox tbname;
        private Panel ptb2;
        private TextBox tbaddress;
        private DataGridView dataGridView;
        private Label ldiscount;
        private CheckBox use;
        private Button bdelete;
    }
}