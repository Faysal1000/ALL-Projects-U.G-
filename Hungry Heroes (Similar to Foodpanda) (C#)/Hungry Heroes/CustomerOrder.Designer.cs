namespace Hungry_Heroes
{
    partial class CustomerOrder
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
            DataGridViewCellStyle dataGridViewCellStyle1 = new DataGridViewCellStyle();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(CustomerOrder));
            display = new DataGridView();
            baddfood = new Button();
            bcheckout = new Button();
            bprevious = new Button();
            bexit = new Button();
            ldisplay = new Label();
            label1 = new Label();
            lquantitydisplay = new Label();
            bplus = new Button();
            bminus = new Button();
            ((System.ComponentModel.ISupportInitialize)display).BeginInit();
            SuspendLayout();
            // 
            // display
            // 
            display.AllowUserToAddRows = false;
            display.AllowUserToDeleteRows = false;
            display.Anchor = AnchorStyles.None;
            display.AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill;
            dataGridViewCellStyle1.Alignment = DataGridViewContentAlignment.MiddleLeft;
            dataGridViewCellStyle1.BackColor = SystemColors.Control;
            dataGridViewCellStyle1.Font = new Font("Segoe UI", 9F, FontStyle.Regular, GraphicsUnit.Point);
            dataGridViewCellStyle1.ForeColor = SystemColors.WindowText;
            dataGridViewCellStyle1.SelectionBackColor = Color.Green;
            dataGridViewCellStyle1.SelectionForeColor = SystemColors.HighlightText;
            dataGridViewCellStyle1.WrapMode = DataGridViewTriState.True;
            display.ColumnHeadersDefaultCellStyle = dataGridViewCellStyle1;
            display.ColumnHeadersHeightSizeMode = DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            display.Location = new Point(12, 12);
            display.MultiSelect = false;
            display.Name = "display";
            display.ReadOnly = true;
            display.RowTemplate.Height = 25;
            display.SelectionMode = DataGridViewSelectionMode.FullRowSelect;
            display.Size = new Size(573, 537);
            display.TabIndex = 0;
            display.CellClick += display_CellClick;
            // 
            // baddfood
            // 
            baddfood.Anchor = AnchorStyles.None;
            baddfood.Cursor = Cursors.Hand;
            baddfood.FlatAppearance.BorderColor = SystemColors.GradientActiveCaption;
            baddfood.FlatAppearance.BorderSize = 3;
            baddfood.FlatAppearance.MouseDownBackColor = Color.FromArgb(0, 192, 0);
            baddfood.FlatAppearance.MouseOverBackColor = Color.Green;
            baddfood.FlatStyle = FlatStyle.Flat;
            baddfood.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            baddfood.Image = Properties.Resources.pngegg__12_;
            baddfood.ImageAlign = ContentAlignment.MiddleLeft;
            baddfood.Location = new Point(589, 191);
            baddfood.Name = "baddfood";
            baddfood.Size = new Size(296, 51);
            baddfood.TabIndex = 1;
            baddfood.Text = "            Add to Food Tray";
            baddfood.TextAlign = ContentAlignment.MiddleLeft;
            baddfood.UseVisualStyleBackColor = true;
            baddfood.Click += baddfood_Click;
            // 
            // bcheckout
            // 
            bcheckout.Anchor = AnchorStyles.None;
            bcheckout.Cursor = Cursors.Hand;
            bcheckout.FlatAppearance.BorderColor = SystemColors.GradientActiveCaption;
            bcheckout.FlatAppearance.BorderSize = 3;
            bcheckout.FlatAppearance.MouseDownBackColor = Color.FromArgb(0, 192, 0);
            bcheckout.FlatAppearance.MouseOverBackColor = Color.Green;
            bcheckout.FlatStyle = FlatStyle.Flat;
            bcheckout.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            bcheckout.Image = (Image)resources.GetObject("bcheckout.Image");
            bcheckout.ImageAlign = ContentAlignment.MiddleLeft;
            bcheckout.Location = new Point(589, 263);
            bcheckout.Name = "bcheckout";
            bcheckout.Size = new Size(296, 51);
            bcheckout.TabIndex = 2;
            bcheckout.Text = "            Check Out Your Tray";
            bcheckout.TextAlign = ContentAlignment.MiddleLeft;
            bcheckout.UseVisualStyleBackColor = true;
            bcheckout.Click += bcheckout_Click;
            // 
            // bprevious
            // 
            bprevious.Anchor = AnchorStyles.None;
            bprevious.Cursor = Cursors.Hand;
            bprevious.FlatAppearance.BorderColor = SystemColors.GradientActiveCaption;
            bprevious.FlatAppearance.BorderSize = 3;
            bprevious.FlatAppearance.MouseDownBackColor = Color.FromArgb(255, 128, 128);
            bprevious.FlatAppearance.MouseOverBackColor = Color.Red;
            bprevious.FlatStyle = FlatStyle.Flat;
            bprevious.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            bprevious.Image = (Image)resources.GetObject("bprevious.Image");
            bprevious.ImageAlign = ContentAlignment.MiddleLeft;
            bprevious.Location = new Point(589, 335);
            bprevious.Name = "bprevious";
            bprevious.Size = new Size(296, 51);
            bprevious.TabIndex = 3;
            bprevious.Text = "            Previous Page";
            bprevious.TextAlign = ContentAlignment.MiddleLeft;
            bprevious.UseVisualStyleBackColor = true;
            bprevious.Click += bprevious_Click;
            // 
            // bexit
            // 
            bexit.Anchor = AnchorStyles.None;
            bexit.Cursor = Cursors.Hand;
            bexit.FlatAppearance.BorderColor = SystemColors.GradientActiveCaption;
            bexit.FlatAppearance.BorderSize = 3;
            bexit.FlatAppearance.MouseDownBackColor = Color.Red;
            bexit.FlatAppearance.MouseOverBackColor = Color.FromArgb(192, 0, 0);
            bexit.FlatStyle = FlatStyle.Flat;
            bexit.Font = new Font("Arial Rounded MT Bold", 14.25F, FontStyle.Regular, GraphicsUnit.Point);
            bexit.Image = (Image)resources.GetObject("bexit.Image");
            bexit.ImageAlign = ContentAlignment.MiddleLeft;
            bexit.Location = new Point(589, 406);
            bexit.Name = "bexit";
            bexit.Size = new Size(296, 51);
            bexit.TabIndex = 4;
            bexit.Text = "              Exit Ordering";
            bexit.TextAlign = ContentAlignment.MiddleLeft;
            bexit.UseVisualStyleBackColor = true;
            bexit.Click += bexit_Click;
            // 
            // ldisplay
            // 
            ldisplay.Anchor = AnchorStyles.None;
            ldisplay.BorderStyle = BorderStyle.Fixed3D;
            ldisplay.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            ldisplay.Location = new Point(593, 15);
            ldisplay.Name = "ldisplay";
            ldisplay.Size = new Size(286, 117);
            ldisplay.TabIndex = 5;
            // 
            // label1
            // 
            label1.Anchor = AnchorStyles.None;
            label1.AutoSize = true;
            label1.Font = new Font("Segoe UI", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            label1.Location = new Point(594, 140);
            label1.Name = "label1";
            label1.Size = new Size(68, 20);
            label1.TabIndex = 6;
            label1.Text = "Quantity:";
            // 
            // lquantitydisplay
            // 
            lquantitydisplay.Anchor = AnchorStyles.None;
            lquantitydisplay.AutoSize = true;
            lquantitydisplay.BorderStyle = BorderStyle.FixedSingle;
            lquantitydisplay.FlatStyle = FlatStyle.Flat;
            lquantitydisplay.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            lquantitydisplay.ForeColor = SystemColors.ActiveCaptionText;
            lquantitydisplay.Location = new Point(662, 140);
            lquantitydisplay.Name = "lquantitydisplay";
            lquantitydisplay.Size = new Size(19, 19);
            lquantitydisplay.TabIndex = 7;
            lquantitydisplay.Text = "1";
            // 
            // bplus
            // 
            bplus.Anchor = AnchorStyles.None;
            bplus.Cursor = Cursors.Hand;
            bplus.FlatAppearance.BorderColor = Color.FromArgb(0, 192, 192);
            bplus.FlatAppearance.MouseOverBackColor = Color.Green;
            bplus.FlatStyle = FlatStyle.Flat;
            bplus.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bplus.Location = new Point(742, 137);
            bplus.Name = "bplus";
            bplus.Size = new Size(27, 26);
            bplus.TabIndex = 8;
            bplus.Text = "+";
            bplus.UseVisualStyleBackColor = true;
            bplus.Click += bplus_Click;
            // 
            // bminus
            // 
            bminus.Anchor = AnchorStyles.None;
            bminus.Cursor = Cursors.Hand;
            bminus.FlatAppearance.BorderColor = Color.FromArgb(0, 192, 192);
            bminus.FlatAppearance.MouseOverBackColor = Color.Red;
            bminus.FlatStyle = FlatStyle.Flat;
            bminus.Font = new Font("Arial Rounded MT Bold", 11.25F, FontStyle.Regular, GraphicsUnit.Point);
            bminus.Location = new Point(712, 137);
            bminus.Name = "bminus";
            bminus.Size = new Size(27, 26);
            bminus.TabIndex = 9;
            bminus.Text = "-";
            bminus.UseVisualStyleBackColor = true;
            bminus.Click += bminus_Click;
            // 
            // CustomerOrder
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.ScrollBar;
            ClientSize = new Size(884, 561);
            Controls.Add(bminus);
            Controls.Add(bplus);
            Controls.Add(lquantitydisplay);
            Controls.Add(label1);
            Controls.Add(ldisplay);
            Controls.Add(bexit);
            Controls.Add(bprevious);
            Controls.Add(bcheckout);
            Controls.Add(baddfood);
            Controls.Add(display);
            Icon = (Icon)resources.GetObject("$this.Icon");
            Name = "CustomerOrder";
            StartPosition = FormStartPosition.CenterScreen;
            Text = "Customer Order";
            Load += CustomerOrder_Load;
            ((System.ComponentModel.ISupportInitialize)display).EndInit();
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private DataGridView display;
        private Button baddfood;
        private Button bcheckout;
        private Button bprevious;
        private Button bexit;
        private Label ldisplay;
        private Label label1;
        private Label lquantitydisplay;
        private Button bplus;
        private Button bminus;
    }
}